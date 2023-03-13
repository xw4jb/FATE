#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
from fate.interface import Dataframe
from fate.interface import Context
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import HeteroSBTGuestSplitter
from fate.ml.ensemble.learner.decision_tree.tree_core.hist_builder import HistogramBuilder, Histogram
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node, DecisionTree

logger = logging.getLogger(__name__)


class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, min_sample_split=2, min_impurity_split=1e-3, l1=0, l2=0, min_child_weight=1, 
                 min_leaf_node=1, use_missing=False, 
                 zero_as_missing=False, feature_importance_type='split', valid_features=None):
        
        """
        Initialize a DecisionTree instance.

        Parameters:
        -----------
        max_depth : int
            The maximum depth of the tree.
        min_sample_split : int
            The minimum number of samples required to split an internal node.
        min_impurity_split : float
            The minimum impurity required to split an internal node.
        min_leaf_node : int
            The minimum number of samples required to be at a leaf node.
        use_missing : bool, optional
            Whether or not to use missing values (default is False).
        zero_as_missing : bool, optional
            Whether to treat zero as a missing value (default is False).
        feature_importance_type : str, optional
            if is 'split', feature_importances calculate by feature split times,
            if is 'gain', feature_importances calculate by feature split gain.
            default: 'split'
            Due to the safety concern, we adjust training strategy of Hetero-SBT in FATE-1.8,
            When running Hetero-SBT, this parameter is now abandoned， guest side will compute split, gain of local features,
            and receive anonymous feature importance results from hosts. Hosts will compute split importance of local features.
        valid_features: list of boolean, optional
            Valid features for training, default is None, which means all features are valid.
        """

        super().__init__(max_depth, use_missing, zero_as_missing, feature_importance_type, valid_features)
        
        # histogram builder
        self.hist_builder = HistogramBuilder()
        self.splitter = HeteroSBTGuestSplitter(l1, l2, min_impurity_split, min_sample_split, min_leaf_node, min_child_weight, 
                                               use_missing=self.use_missing)

    def _encrypt_grad_and_hess(self, g_tensor, h_tensor, encryptor):
        en_g = encryptor.encrypt(g_tensor)
        en_h = encryptor.encrypt(h_tensor)
        return en_g, en_h
    
    def _mask_node_list(self, node_list):
        new_node_list = []
        for node in node_list:
            new_node_list.append(Node(nid=node.nid, is_left_node=node.is_left_node))
        return new_node_list
    
    def _make_gh_sum_tensor(self, node_list):
        pass

    def fit(self, ctx: Context, train_data: Dataframe, grad_and_hess: Dataframe, encryptor):
        
        logger.info('start to fit hetero decision tree guest')

        # initialize root node
        self._g_tensor, self._h_tensor =  grad_and_hess.g.as_tensor(), grad_and_hess.h.as_tensor()
        root_node = self._initialize_root_node(self._g_tensor, self._h_tensor, None)
        self._cur_layer_node.append(root_node)

        # process grad and hess
        en_g, en_h = self._encrypt_grad_and_hess(self._g_tensor, self._h_tensor, encryptor)
        ctx.hosts.put('grad_and_hess', [en_g, en_h])

        # initialize positions
        self._sample_pos = self._init_sample_pos(train_data)

        # build tree
        for depth, layer_ctx in ctx.range(self.max_depth):
            
            logger.info('start to fit hetero decision tree guest, depth: {}'.format(depth))
            layer_ctx.hosts.put('node_to_split', self._mask_node_list(self._cur_layer_node))
            # compute histogram
            hist: Histogram = self.hist_builder.compute_histogram(train_data, gh=grad_and_hess, pos_df=self._sample_pos, 
                                                                  cur_layer_node=self._cur_layer_node, valid_features=self._valid_feature, 
                                                                  use_missing=self.use_missing, zero_as_missing=self.zero_as_missing)
            # get encrypted split points from host
            host_enc_split_points = layer_ctx.hosts.get('enc_split_points')
            
            splits, host_splits = self.splitter.federated_split(guest_histogram=hist, host_splits=host_enc_split_points, 
                                                                cur_layer_node=self._cur_layer_node, encryptor=encryptor)
            
            # update node pos