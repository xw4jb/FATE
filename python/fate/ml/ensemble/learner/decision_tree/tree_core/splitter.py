from typing import List
import torch as t
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node
from fate.ml.ensemble.learner.decision_tree.tree_core.hist_builder import Histogram

def convert_memory_list_to_tensor(list_):
    return list_


class SplitInfo(object):
    def __init__(self, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, mask_id=None, sample_count=-1):
        
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir
        self.mask_id = mask_id
        self.sample_count = sample_count

    def __repr__(self):
        return '(fid {} bid {}, sum_grad {}, sum_hess {}, gain {}, sitename {}, missing dir {}, mask_id {}, ' \
        'sample_count {})\n'.format(
            self.best_fid, self.best_bid, self.sum_grad, self.sum_hess, self.gain, self.sitename, self.missing_dir,
            self.mask_id, self.sample_count)


class XgboostSplitter(object):

    def __init__(self, feature_num, feature_bin_num: List[int], 
                 l1=0, l2=0, min_impurity_split=1e-2, min_sample_split=2, min_leaf_node=1, min_child_weight=1):
        
        self.feature_num = feature_num
        self.feature_bin_num = feature_bin_num
        self.l1 = l1
        self.l2 = l2
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight

    def index_to_fid_bid(self, index, use_missing):
        pass

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

    def _make_split_info(self, best_split, ):
        pass

    def split(self, histogram: Histogram, cur_layer_node: List[Node], use_missing: bool):
        
        g_list = [node.grad for node in cur_layer_node]
        h_list = [node.hess for node in cur_layer_node]
        cnt_list = [node.sample_num for node in cur_layer_node]
        g_sum = convert_memory_list_to_tensor(g_list)
        h_sum = convert_memory_list_to_tensor(h_list)
        cnt_sum = convert_memory_list_to_tensor(cnt_list)

        g_l = histogram[:-1].g_hist.flatten()
        h_l = histogram[:-1].h_hist.flatten()
        cnt_l = histogram[:-1].cnt.flatten()

        if use_missing:
            g_l_miss = (histogram.g_hist + histogram.missing_g)[:-1].flatten()
            h_l_miss = (histogram.h_hist + histogram.missing_h)[:-1].flatten()
            cnt_l_miss = (histogram.cnt + histogram.missing_count)[:-1].flatten()
            g_l = g_l.vstack([g_l, g_l_miss])
            h_l = h_l.vstack([h_l, h_l_miss])
            cnt_l = cnt_l.vstack([cnt_l, cnt_l_miss])

        g_r = g_sum - g_l
        h_r = h_sum - h_l
        cnt_r = cnt_sum - cnt_l
        gain = self._gain(g_l, h_l, g_r, h_r, g_sum, h_sum)

        mask = self._condition_mask(gain, cnt_l, cnt_r, g_l, g_r)
        gain[mask] = - t.inf  # these split point are invalid
        best_gain = gain.max()
        best_split = gain.argmax()

        node_mask = self._node_min_sample_split_mask(cnt_sum)
        best_gain[node_mask] = -t.inf

        # mask node that can not be split
        return best_gain, best_split

    def make_enc_splits(self, encryptor):
        pass
        
    def compute_enc_split(self, encryptor):
        pass

    def find_best_splits_among_parties(self,):
        pass