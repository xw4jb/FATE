from typing import List
import torch as t
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node
from fate.ml.ensemble.learner.decision_tree.tree_core.hist_builder import Histogram

def convert_memory_list_to_tensor(list_):
    return list_


class SplitInfo(object):
    def __init__(self, node_id=None, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, mask_id=None, sample_count=-1):
        
        self.node_id = node_id
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir
        self.mask_id = mask_id
        self.sample_count = sample_count

    def __repr__(self):
        return '(nid {} fid {} bid {}, sum_grad {}, sum_hess {}, gain {}, sitename {}, missing dir {}, mask_id {}, ' \
        'sample_count {})\n'.format(self.node_id,
            self.best_fid, self.best_bid, self.sum_grad, self.sum_hess, self.gain, self.sitename, self.missing_dir,
            self.mask_id, self.sample_count)


class XgboostSplitter(object):

    def __init__(self, feature_num, feature_bin_num: List[int], 
                 l1=0, l2=0, min_impurity_split=1e-2, min_sample_split=2, min_leaf_node=1, min_child_weight=1, use_missing=False):
        
        self.feature_num = feature_num
        self.feature_bin_num = feature_bin_num
        assert self.feature_num == len(feature_bin_num), 'feature number not match'
        self.l1 = l1
        self.l2 = l2
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight
        self.index_to_split_info_map = {}
        self.use_missing = use_missing

        # preprocess
        self._map_index_to_fid_bid()

    def _map_index_to_fid_bid(self):
        split_idx = 0
        for fid, bin_num in enumerate(self.feature_bin_num):
            for bid in range(bin_num - 1):
                self.index_to_split_info_map[split_idx] = (fid, bid, True) # fid, bid, missing_dir
                split_idx += 1

        if self.use_missing:
            missing_dict = {}
            split_idx = len(self.index_to_split_info_map)
            for k, v in self.index_to_split_info_map.items():
                missing_dict[split_idx] = (v[0], v[1], False)
                split_idx += 1
            self.index_to_split_info_map.update(missing_dict)

    def _apply_l1(self, g):
        
        g = g[g < -self.l1] + self.l1
        g = g[g > self.l1] - self.l1
        g[(g <= self.l1) & (g >= -self.l1)] = 0

        return g

    def _gain(self, g_l, h_l, g_r, h_r, g_sum, h_sum):
        
        if self.l1 != 0:  # l1 regularization
            g_l, g_r, g_sum = self._apply_l1(g_l), self._apply_l1(g_r), self._apply_l1(g_sum)

        lambd_ = self.l2  # l2 regularization
        gain = (g_l ** 2 / (h_l + lambd_)) + (g_r ** 2 / (h_r + lambd_)) - (g_sum ** 2 / (h_sum + lambd_))

        return gain
    
    def _condition_mask(self, gain, cnt_l, cnt_r, g_l, g_r): 

        # compute min impurity split mask
        mask_0 = gain < self.min_impurity_split

        # compute min sample split mask
        mask_1 = cnt_l < self.min_leaf_node
        mask_2 = cnt_r < self.min_leaf_node

        # compute min child weight mask
        mask_3 = g_l < self.min_child_weight
        mask_4 = g_r < self.min_child_weight

        mask = mask_0 & mask_1 & mask_2 & mask_3 & mask_4  # if is True in mask, this split is invalid

        return mask

    def _node_min_sample_split_mask(self, cnt_sum):
        return cnt_sum > self.min_sample_split  # if is True in mask, this node can not be split

    def _get_split_info(self, best_split_idx, best_gain, g_l, h_l, cnt_l, node_id_map):

        split_points = []
        best_g = g_l[best_split_idx]
        best_h = h_l[best_split_idx]
        best_cnt = cnt_l[best_split_idx]
        idx = 0
        for split_idx, gain, g, h, cnt in zip(best_split_idx, best_gain, best_g, best_h, best_cnt):
            if gain == -t.inf:
                split_points.append(None)
            fid, bid, missing_dir = self.index_to_split_info_map[split_idx]
            split_info = SplitInfo(node_id_map[idx], fid, bid, g, h, gain, missing_dir, sample_count=cnt)
            split_points.append(split_info)
            idx += 1

        return split_points
    
    def _compute_node_ghcnt_sum(self, cur_layer_node):
        
        g_list = [node.grad for node in cur_layer_node]
        h_list = [node.hess for node in cur_layer_node]
        cnt_list = [node.sample_num for node in cur_layer_node]
        g_sum = convert_memory_list_to_tensor(g_list)
        h_sum = convert_memory_list_to_tensor(h_list)
        cnt_sum = convert_memory_list_to_tensor(cnt_list)
        return g_sum, h_sum, cnt_sum

    def _process_histogram(self, histogram: Histogram):
        
        g_l = histogram[:-1].g_hist.flatten()
        h_l = histogram[:-1].h_hist.flatten()
        cnt_l = histogram[:-1].cnt.flatten()

        if self.use_missing:
            g_l_miss = (histogram.g_hist + histogram.missing_g)[:-1].flatten()
            h_l_miss = (histogram.h_hist + histogram.missing_h)[:-1].flatten()
            cnt_l_miss = (histogram.cnt + histogram.missing_count)[:-1].flatten()
            g_l = g_l.vstack([g_l, g_l_miss])
            h_l = h_l.vstack([h_l, h_l_miss])
            cnt_l = cnt_l.vstack([cnt_l, cnt_l_miss])

        return g_l, h_l, cnt_l
    
    def _compute_best_split(self, g_l, h_l, cnt_l, g_r, h_r, cnt_r, g_sum, h_sum):
        
        gain = self._gain(g_l, h_l, g_r, h_r, g_sum, h_sum)
        mask = self._condition_mask(gain, cnt_l, cnt_r, g_l, g_r)
        gain[mask] = - t.inf  # these split point are invalid
        best_gain = gain.max()
        best_split_idx = gain.argmax()

        return best_split_idx, best_gain

    def split(self, histogram: Histogram, cur_layer_node: List[Node]):
        
        g_l, h_l, cnt_l = self._prepare_for_split(histogram)
        g_sum, h_sum, cnt_sum = self._compute_node_ghcnt_sum(cur_layer_node)

        g_r = g_sum - g_l
        h_r = h_sum - h_l
        cnt_r = cnt_sum - cnt_l

        best_split_idx, best_gain = self._compute_best_split(g_l, h_l, cnt_l, g_r, h_r, cnt_r, g_sum, h_sum)
        node_mask = self._node_min_sample_split_mask(cnt_sum)
        best_gain[node_mask] = -t.inf
        node_id_map = {idx: node.nid for idx, node in enumerate(cur_layer_node)}
        split_points = self._get_split_info(best_split_idx, best_gain, g_l, h_l, cnt_l, node_id_map)

        return split_points


class HeteroSBTGuestSplitter(XgboostSplitter):

    def __init__(self, feature_num, feature_bin_num: List[int], l1=0, l2=0, 
                 min_impurity_split=0.01, min_sample_split=2, min_leaf_node=1, 
                 min_child_weight=1, use_missing=False):
        super().__init__(feature_num, feature_bin_num, l1, l2, min_impurity_split, min_sample_split, min_leaf_node, min_child_weight, use_missing)

        self._guest_split_num = len(self.index_to_split_info_map)
        self._host_shuffle_idx = None

    def _shuffle_host_splits(self, g_l, h_l, cnt_l):
        g_l, index = g_l.shuffle()
        h_l = h_l[index]
        cnt_l = cnt_l[index]
        return g_l, h_l, cnt_l, index
    
    def _get_host_split_idx(self, split_idx, host_split_num):
        
        offset = split_idx - self._guest_split_num
        host_idx = 0
        for n_split in host_split_num:
            if offset > n_split:
                offset -= n_split
                host_idx += 1
            else:
                break
        
        host_split_idx = offset
        return host_idx, host_split_idx
    
    def _get_split_info(self, best_split_idx, best_gain, g_l, h_l, cnt_l, host_split_num, node_id_map):

        split_points = []
        host_splits = [[] for i in range(host_split_num)]
        best_g = g_l[best_split_idx]
        best_h = h_l[best_split_idx]
        best_cnt = cnt_l[best_split_idx]
        idx = 0
        for split_idx, gain, g, h, cnt in zip(best_split_idx, best_gain, best_g, best_h, best_cnt):
            if gain == -t.inf:
                split_points.append(None)
            if split_idx < self._guest_split_num:
                fid, bid, missing_dir = self.index_to_split_info_map[split_idx]
                split_info = SplitInfo(node_id_map[idx], fid, bid, g, h, gain, missing_dir, sample_count=cnt)
                split_points.append(split_info)
            else:
                # guest save this split point locally
                split_points.append(SplitInfo(node_id_map[idx], sum_grad=g, sum_hess=h, gain=gain, sample_count=cnt))
                # this split point belongs to host, save it in host_splits and will then be sent to host 
                host_idx, host_split_idx = self._get_host_split_idx(split_idx, host_split_num)
                host_splits[host_idx].append(SplitInfo(node_id_map[idx], mask_id=host_split_idx))

        return split_points, host_splits

    def federated_split(self, guest_histogram, host_splits, cur_layer_node, encryptor):
        
        g_l, h_l, cnt_l = self._prepare_for_split(guest_histogram)
        g_sum, h_sum, cnt_sum = self._compute_node_ghcnt_sum(cur_layer_node)

        host_split_num = []
        for host_data in host_splits: 
            # merge split-info
            host_g_l, host_h_l, host_cnt_l = host_data
            host_g_l = encryptor.decrypt(host_g_l)
            host_h_l = encryptor.decrypt(host_h_l)
            host_split_num.append(host_g_l.shape[0])
            g_l = g_l.vstack([g_l, host_g_l])
            h_l = h_l.vstack([h_l, host_h_l])
            cnt_l = cnt_l.vstack([cnt_l, host_cnt_l])
        
        g_r = g_sum - g_l
        h_r = h_sum - h_l
        cnt_r = cnt_sum - cnt_l

        best_split_idx, best_gain = self._compute_best_split(g_l, h_l, cnt_l, g_r, h_r, cnt_r, g_sum, h_sum)
        node_mask = self._node_min_sample_split_mask(cnt_sum)
        best_gain[node_mask] = -t.inf
        node_id_map = {idx: node.nid for idx, node in enumerate(cur_layer_node)}
        split_points, host_best_splits = self._get_split_info(best_split_idx, best_gain, g_l, h_l, cnt_l, host_split_num, node_id_map)

        return split_points, host_best_splits


class HeteroSBTHostSplitter(XgboostSplitter):

    def __init__(self, feature_num, feature_bin_num: List[int], l1=0, l2=0, 
                 min_impurity_split=0.01, min_sample_split=2, min_leaf_node=1, 
                 min_child_weight=1, use_missing=False):
        super().__init__(feature_num, feature_bin_num, l1, l2, min_impurity_split, min_sample_split, min_leaf_node, min_child_weight, use_missing)
        self._host_shuffle_idx = None

    def host_make_enc_splits(self, histogram: Histogram):
        # encrypted 
        g_l, h_l, cnt_l = self._prepare_for_split(histogram)
        # shuffle to mess the split points
        g_l, h_l, cnt_l, index = self._shuffle_host_splits(g_l, h_l, cnt_l)
        self._host_shuffle_idx = index.to_local()

        return g_l, h_l, cnt_l
    
    def _host_reconstruct_split_info(self, host_best_splits):
        pass


if __name__ == '__main__':
    splitter = XgboostSplitter(5, [2, 4, 3, 2, 3], use_missing=True)