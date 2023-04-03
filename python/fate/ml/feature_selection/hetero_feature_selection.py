#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

import copy
import json
import logging
import random

import numpy as np
import pandas as pd

from fate.interface import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class HeteroSelectionModuleGuest(Module):
    def __init__(self, method, isometric_model_dict=None,
                 iv_param=None, statistic_param=None, manual_param=None,
                 keep_one=True):
        self.method = method
        self.isometric_model_dict = isometric_model_dict
        self.iv_param = iv_param
        self.statistic_param = statistic_param
        self.manual_param = manual_param
        self.keep_one = keep_one
        # for display of cascade order
        self._inner_method = [None] * len(method)
        self._selection_obj = [None] * len(method)

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        header = train_data.schema.columns
        for i, filter_type in enumerate(self.method):
            if filter_type == "manual":
                selection_obj = ManualSelection(method=self.method,
                                                header=header,
                                                param=self.manual_param,
                                                keep_one=self.keep_one)
                self._selection_obj[-1] = selection_obj
                self._inner_method[-1] = "manual"
            elif filter_type == "iv":
                model = self.isometric_model_dict.get("binning", None)
                if model is None:
                    raise ValueError(f"Cannot find binning model in input, please check")
                selection_obj = StandardSelection(method=self.method,
                                                  header=header,
                                                  param=self.iv_param,
                                                  model=model,
                                                  keep_one=self.keep_one)
                self._selection_obj[i] = selection_obj
                self._inner_method[i] = "iv"
            elif filter_type == "statistic":
                model = self.isometric_model_dict.get("statistic", None)
                if model is None:
                    raise ValueError(f"Cannot find statistic model in input, please check")
                selection_obj = StandardSelection(method=self.method,
                                                  header=header,
                                                  param=self.statistic_param,
                                                  model=model,
                                                  keep_one=self.keep_one)
                self._selection_obj[i] = selection_obj
                self._inner_method[i] = "statistic"
            else:
                raise ValueError(f"{filter_type} selection method not supported, please check")

        prev_selection_obj = None
        for method, selection_obj in zip(self._inner_method, self._selection_obj):
            if prev_selection_obj:
                selection_obj.set_prev_selected_mask(copy.deepcopy(prev_selection_obj._selected_mask))
                if isinstance(selection_obj, StandardSelection) and isinstance(prev_selection_obj, StandardSelection):
                    selection_obj.set_host_prev_selected_mask(copy.deepcopy(prev_selection_obj._host_selected_mask))
            selection_obj.fit(ctx, train_data, validate_data)
            if method == "binning":
                if self.iv_param.select_federated:
                    self.sync_select_federated(ctx, selection_obj)
            prev_selection_obj = selection_obj

    def sync_select_federated(self, ctx: Context, selection_obj):
        logger.info(f"Sync federated selection.")
        for i, host in enumerate(ctx.hosts):
            federated_mask = selection_obj._host_selected_mask[host]
            ctx.hosts[i].put(f"selected_mask_{selection_obj.method}", federated_mask)

    def transform(self, ctx: Context, test_data):
        transformed_data = self._selection_obj[-1].transform(ctx, test_data)
        return transformed_data

    def to_model(self):
        # all selection obj need to be recorded for display of cascade order
        selection_obj_list = []
        for selection_obj in self._selection_obj:
            selection_obj_list.append(selection_obj.to_model())
        return {"selection_obj_list": json.dumps(selection_obj_list)}

    def restore(self, model):
        selection_obj_list = []
        selection_obj_model_list = json.loads(model["selection_obj_list"])
        for i, selection_model in enumerate(selection_obj_model_list):
            selection_obj = StandardSelection(self._inner_method[i])
            selection_obj.restore(selection_model)
            selection_obj_list.append(selection_obj)
        self._selection_obj = selection_obj_list

    @classmethod
    def from_model(cls, model) -> "HeteroSelectionModuleGuest":
        selection_obj = HeteroSelectionModuleGuest(model["method"])
        selection_obj._inner_method = model["_inner_method"]
        selection_obj.restore(model)
        return selection_obj


class HeteroSelectionModuleHost(Module):
    def __init__(self, method, isometric_model_dict=None,
                 iv_param=None, statistic_param=None, manual_param=None,
                 keep_one=True):
        self.method = method
        self.isometric_model_dict = isometric_model_dict
        self.iv_param = iv_param
        self.statistic_param = statistic_param
        self.manual_param = manual_param
        self.keep_one = keep_one
        # for display of cascade order
        self._inner_method = [None] * len(method)
        self._selection_obj = [None] * len(method)

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        header = train_data.schema.columns
        for i, type in enumerate(self.method):
            if type == "manual":
                selection_obj = ManualSelection(method=self.method,
                                                header=header,
                                                param=self.manual_param,
                                                keep_one=self.keep_one)
                self._selection_obj[-1] = selection_obj
                self._inner_method[-1] = "manual"
            elif type == "iv":
                model = self.isometric_model_dict["binning"]
                selection_obj = StandardSelection(method=self.method,
                                                  header=header,
                                                  param=self.iv_param,
                                                  model=model,
                                                  keep_one=self.keep_one)
                self._selection_obj[i] = selection_obj
                self._inner_method[i] = "iv"
            elif type == "statistic":
                model = self.isometric_model_dict["statistic"]
                selection_obj = StandardSelection(method=self.method,
                                                  header=header,
                                                  param=self.statistic_param,
                                                  model=model,
                                                  keep_one=self.keep_one)
                self._selection_obj[i] = selection_obj
                self._inner_method[i] = "statistic"
            else:
                raise ValueError(f"{type} selection method not supported, please check")

        prev_selection_obj = None
        for method, selection_obj in zip(self._inner_method, self._selection_obj):
            if prev_selection_obj:
                selection_obj.set_prev_selected_mask(copy.deepcopy(prev_selection_obj._selected_mask))
            selection_obj.fit(ctx, train_data, validate_data)
            if method == "binning":
                if self.iv_param.select_federated:
                    self.sync_select_federated(ctx, selection_obj)
            prev_selection_obj = selection_obj

    def sync_select_federated(self, ctx: Context, selection_obj):
        selected_mask = ctx.guest.get(f"selected_mask_{selection_obj.method}")
        selection_obj.set_selected_mask(selected_mask)

    def transform(self, ctx: Context, test_data):
        transformed_data = self._selection_obj[-1].transform(ctx, test_data)
        return transformed_data

    def to_model(self):
        # all selection obj need to be recorded for display of cascade order
        selection_obj_list = []
        for selection_obj in self._selection_obj:
            selection_obj_list.append(selection_obj.to_model())
        return {"selection_obj_list": json.dumps(selection_obj_list)}

    def restore(self, model):
        selection_obj_list = []
        selection_obj_model_list = json.loads(model["selection_obj_list"])
        for i, selection_model in enumerate(selection_obj_model_list):
            selection_obj = StandardSelection(self._inner_method[i])
            selection_obj.restore(selection_model)
            selection_obj_list.append(selection_obj)
        self._selection_obj = selection_obj_list

    @classmethod
    def from_model(cls, model) -> "HeteroSelectionModuleHost":
        selection_obj = HeteroSelectionModuleHost(model["method"])
        selection_obj._inner_method = model["_inner_method"]
        selection_obj.restore(model)
        return selection_obj


class ManualSelection(Module):
    def __init__(self, method, param=None, header=None, model=None, keep_one=True):
        assert method == "manual", f"only 'manual' is accepted, received {method} instead."
        self.method = method
        self.param = param
        self.model = model
        self.keep_one = keep_one
        self._header = header
        self._prev_selected_mask = None
        if header is None:
            self._selected_mask = None
        else:
            self._selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)

    def set_selected_mask(self, mask):
        self._selected_mask = mask

    def set_prev_selected_mask(self, mask):
        self._prev_selected_mask = mask

    def fit(self, ctx: Context, train_data, validate_data=None):
        header = train_data.schema.columns
        if self._header is None:
            self._header = header
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)

        filter_out_col = self.param["filter_out_col"]
        keep_col = self.param["keep_col"]
        if len(filter_out_col) >= len(header):
            raise ValueError("`filter_out_col` should not be all columns")
        if filter_out_col is None:
            filter_out_col = []
        if keep_col is None:
            keep_col = []
        filter_out_col = set(filter_out_col)
        keep_col = set(keep_col)
        filter_out_mask = [0 if col in filter_out_col else 1 for col in self._header]
        keep_mask = [1 if col in keep_col else 0 for col in self._header]
        selected_mask = self._prev_selected_mask & filter_out_mask
        selected_mask += keep_mask
        self._selected_mask = selected_mask > 0
        if self.keep_one:
            StandardSelection._keep_one(self._selected_mask, self._prev_selected_mask)

    def transform(self, ctx: Context, transform_data):
        logger.debug(f"Start transform")
        select_cols = [col for col, mask in zip(self._header, self._selected_mask) if mask]
        return transform_data[select_cols]

    def to_model(self):
        return dict(
            method=self.method,
            keep_one=self.keep_one,
            selected_mask=self._selected_mask.to_dict()
        )

    def restore(self, model):
        self.method = model["method"]
        self.keep_one = model["keep_one"]
        self._selected_mask = pd.Series(["selected_mask"])


class StandardSelection(Module):
    def __init__(self, header, method, param=None, model=None, keep_one=True):
        self.method = method
        self.param = param
        self.filter_conf = {}
        for metric_name, filter_type, threshold, high_take in zip(self.param["metrics"],
                                                                  self.param["filter_type",
                                                                  self.param["threshold"],
                                                                  self.param["take_high"]]):
            metric_conf = self.filter_conf.get(metric_name, {})
            metric_conf["filter_type"] = metric_conf.get("filter_type", []) + [filter_type]
            metric_conf["threshold"] = metric_conf.get("threshold", []) + [threshold]
            metric_conf["take_high"] = metric_conf.get("take_high", []) + [high_take]
        self.model = self.convert_model(model)
        self.keep_one = keep_one
        self._header = header
        self._selected_mask = None
        self._all_selected_mask = None
        if header is None:
            self._prev_selected_mask = None
        else:
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)
        self._host_selected_mask = {}
        self._all_host_selected_mask = {}
        self._host_prev_selected_mask = {}
        self._all_metrics = None
        self._all_host_metrics = None

    @staticmethod
    def convert_model(input_model):
        return input_model

    def set_host_prev_selected_mask(self, mask):
        self._host_prev_selected_mask = mask

    def set_prev_selected_mask(self, mask):
        self._prev_selected_mask = mask

    def fit(self, ctx: Context, train_data, validate_data=None):
        if self._header is None:
            header = train_data.schema.columns
            self._header = header
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)
        """if self.method == "manual":
            filter_out_col = self.param["filter_out_col"]
            keep_col = self.param["keep_col"]
            if filter_out_col is None:
                filter_out_col = []
            if keep_col is None:
                keep_col = []
            filter_out_col = set(filter_out_col)
            keep_col = set(keep_col)
            filter_out_mask = [0 if col in filter_out_col else 1 for col in self._header]
            keep_mask = [1 if col in keep_col else 0 for col in self._header]
            self._selected_mask *= filter_out_mask
            self._selected_mask += keep_mask
            self._selected_mask = self._selected_mask > 0
            if self.keep_one:
                self._keep_one()
        """
        metric_names = self.param.metrics
        # local only
        if self.method in ["statistic"]:
            for metric_name in metric_names:
                if metric_name not in self.model["metrics"]:
                    raise ValueError(f"metric {metric_name} not found in given statistic model with metrics: "
                                     f"{metric_names}, please check")

            metrics_all = pd.DataFrame(self.model["metrics_summary"]).loc[metric_names]
            self._all_metrics = metrics_all
            """ mask_all = metrics_all.apply(lambda r: StandardSelection.filter_multiple_metrics(r,
                                                                                             self.param.filter_type,
                                                                                             self.param.threshold,
                                                                                             self.param.take_high,
                                                                                             metric_names), axis=1)"""
            mask_all = self.apply_filter(metrics_all, self.filter_conf)
            self._all_selected_mask = mask_all
            self._selected_mask = self._prev_selected_mask & mask_all.all(axis=0)
            if self.keep_one:
                self._keep_one(self._selected_mask, self._prev_selected_mask)
        # federated selection possible
        elif self.method == "iv":
            iv_metrics = pd.Series(self.model["metrics_summary"]["iv"])
            metrics_all = pd.DataFrame(iv_metrics).T.rename({0: "iv"}, axis=0)
            self._all_metrics = metrics_all
            # works for multiple iv filters
            """mask_all = metrics_all.apply(lambda r: StandardSelection.filter_multiple_metrics(r,
                                                                                             self.param.filter_type,
                                                                                             self.param.threshold,
                                                                                             self.param.take_high,
                                                                                             metric_names), axis=1)
            """
            mask_all = self.apply_filter(metrics_all, self.filter_conf)
            self._all_selected_mask = mask_all
            self._selected_mask = self._prev_selected_mask & mask_all.all(axis=0)
            if self.keep_one:
                self._keep_one(self._selected_mask, self._prev_selected_mask)
            if self.param.select_federated:
                host_metrics_summary = self.model["host_train_metrics_summary"]
                for host, host_metrics in host_metrics_summary.items():
                    iv_metrics = pd.Series(host_metrics["iv"])
                    metrics_all = pd.DataFrame(iv_metrics).T.rename({0: "iv"}, axis=0)
                    """host_mask_all = metrics_all.apply(lambda r:
                                                 StandardSelection.filter_multiple_metrics(r,
                                                                                           self.param.host_filter_type,
                                                                                                     self.param.threshold,
                                                                                                     self.param.take_high,
                                                                                                     metric_names), axis=1)
                    """
                    host_mask_all = self.apply_filter(metrics_all,
                                                      self.filter_conf)
                    self._all_host_selected_mask = host_mask_all
                    host_prev_selected_mask = self._host_prev_selected_mask.get(host)
                    if host_prev_selected_mask is None:
                        host_prev_selected_mask = pd.Series(np.ones(len(iv_metrics.index)),
                                                            index=iv_metrics.index)
                        self._host_prev_selected_mask[host] = host_prev_selected_mask
                    host_selected_mask = self._host_prev_selected_mask.get(host) & host_mask_all.all(axis=0)
                    if self.keep_one:
                        self._keep_one(host_selected_mask,
                                       host_prev_selected_mask)
                    self._host_selected_mask[host] = host_selected_mask

    @staticmethod
    def _keep_one(cur_mask=None, last_mask=None):
        if last_mask is None:
            return cur_mask
        if sum(cur_mask) > 0:
            return cur_mask
        else:
            choice_mask = last_mask.index[last_mask]
            cur_mask[random.choice(choice_mask.index)] = True

    @staticmethod
    def convert_series_metric_to_dataframe(metrics, metric_name):
        return pd.DataFrame(metrics).T.rename({0: metric_name}, axis=0)

    @staticmethod
    def apply_filter(metrics_all, filter_conf):
        return metrics_all.apply(lambda r:
                                 StandardSelection.filter_multiple_metrics(r,
                                                                           filter_conf[r.name]),
                                 axis=1)

    @staticmethod
    def filter_multiple_metrics(metrics, metric_conf):
        filter_type_list = metric_conf["filter_type"]
        threshold_list = metric_conf["threshold"]
        take_high_list = metric_conf["take_high"]
        result = pd.Series(np.ones(len(metrics.index)), index=metrics.index, dtype=bool)
        for idx, method in enumerate(filter_type_list):
            result &= StandardSelection.filter_metrics(metrics, method, threshold_list[idx], take_high_list[idx])
        return result

    @staticmethod
    def filter_metrics(metrics, method, threshold, take_high=True):
        if method == "top_k":
            return StandardSelection.filter_by_top_k(metrics, threshold, take_high)
        elif method == "threshold":
            return StandardSelection.filter_by_threshold(metrics, threshold, take_high)
        elif method == "percentile":
            return StandardSelection.filter_by_percentile(metrics, threshold, take_high)
        else:
            raise ValueError(f"method {method} not supported, please check")

    @staticmethod
    def filter_by_top_k(metrics, k, take_high=True):
        # strict top k
        if k == 0:
            return pd.Series(np.ones(len(metrics)), dtype=bool)
        # stable sort
        ordered_metrics = metrics.sort_values(ascending=~take_high, kind="mergesort")
        select_k = ordered_metrics.index[:k].index
        return metrics.index.isin(select_k)

    @staticmethod
    def filter_by_threshold(metrics, threshold, take_high=True):
        if take_high:
            return metrics >= threshold
        else:
            return metrics <= threshold

    @staticmethod
    def filter_by_percentile(metrics, percentile, take_high=True):
        if take_high:
            return metrics >= metrics.quantile(percentile)
        else:
            return metrics <= metrics.quantile(1 - percentile)

    def transform(self, ctx: Context, transform_data):
        logger.debug(f"Start transform")
        select_cols = self._selected_mask[self._selected_mask]
        return transform_data[select_cols]

    def to_model(self):
        return dict(
            method=self.method,
            keep_one=self.keep_one,
            all_selected_mask=self._all_selected_mask.to_dict(),
            all_metrics=self._all_metrics.to_dict(),
            all_host_metrics=self._all_host_metrics.to_dict(),
            selected_mask=self._selected_mask.to_dict(),
            host_selected_mask={k: v.to_dict() for k, v in self._host_selected_mask.items()},
            all_host_selected_mask={k: v.to_dict() for k, v in self._all_host_selected_mask.items()},
        )

    def restore(self, model):
        self.method = model["method"]
        self.keep_one = model["keep_one"]
        self._selected_mask = pd.Series(["selected_mask"])
        self._all_selected_mask = pd.DataFrame(model["all_selected_mask"])
        self._all_metrics = pd.DataFrame(model["all_metrics"])
        self._host_selected_mask = {k: pd.Series(v) for k, v in model["host_selected_mask"].items()}
        self._all_host_selected_mask = pd.DataFrame(model["all_host_selected_mask"])
        self._all_host_metrics = pd.DataFrame(model["all_host_metrics"])
