from collections import OrderedDict
from typing import List
from fate.interface import Dataframe
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node

def sbt_hist(train_data, gh, use_missing, pos_df, zero_as_missing, valid_features):
    pass

def recombine_histogram(hist_a, hist_b, hist_a_node_order, hist_b_node_order):
    pass


class Histogram(object):

    def __init__(self, g_hist, h_hist, cnt, missing_g, missing_h, missing_cnt) -> None:
        self.g_hist = g_hist
        self.h_hist = h_hist
        self.cnt = cnt
        self.missing_g = missing_g
        self.missing_h = missing_h
        self.missing_cnt = missing_cnt

    def __sub__(self, other):
        
        g_sib = self.g_hist - other.g_hist
        h_sib = self.h_hist - other.h_hist
        cnt_sib = self.cnt - other.cnt
        missing_g_sib = self.missing_g - other.missing_g
        missing_h_sib = self.missing_h - other.missing_h
        missing_cnt_sib = self.missing_cnt - other.missing_cnt

        return Histogram(g_sib, h_sib, cnt_sib, missing_g_sib, missing_h_sib, missing_cnt_sib)


class HistogramBuilder(object):

    def __init__(self,) -> None:

        self._cached_hist = None

    def compute_histogram(self, train_data: Dataframe, gh: Dataframe, pos_df: Dataframe, cur_layer_node: List[Node], 
                                valid_features, use_missing, zero_as_missing):
        
        selected_node = OrderedDict()
        hist_sub = True

        if len(cur_layer_node) == 1 and cur_layer_node[0].nid == 0:
            selected_node[0] = True
            hist_sub = False
        else:
            for left, right in zip(cur_layer_node[::2], cur_layer_node[1::2]):
                # compute the node with less samples
                selected_node[left.nid], selected_node[right.nid] = True, False
                if left.sample_num >= right.sample_num:
                    selected_node[left.nid] = False
                    selected_node[right.nid] = True

            node_to_compute = [node.nid for node in cur_layer_node if selected_node[node.nid]]
            node_left = [node.nid for node in cur_layer_node if not selected_node[node.nid]]
            
            select_idx = train_data.create_frame()
            select_idx['idx'] = pos_df['node_idx'].apply_map_dict(selected_node)
            train_data = train_data[select_idx.idx]
            gh = gh[select_idx.idx]
            pos_df = pos_df[select_idx.idx]

        # compute histogram
        g_hist, h_hist, cnt, missing_g, missing_h, missing_cnt = sbt_hist(train_data, gh, pos_df, use_missing, 
                                                                          zero_as_missing, valid_features)
        hist = Histogram(g_hist, h_hist, cnt, missing_g, missing_h, missing_cnt)
        # histogram subtraction
        if hist_sub:
            sibling_node_hist = self._cached_hist - hist
            # recombine histogram
            hist = recombine_histogram(hist, sibling_node_hist, node_to_compute, node_left)
        # store histogram
        self._cached_hist = hist
        
        return hist
    