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
from sklearn import metrics
from fate.ml.evaluation.metrics.metric_base import Metric
from fate.ml.evaluation.metric_list import REGRESSION_METRICS, BINARY_METRICS, MULTICLASS_METRICS


class Evaluator(object):

    def __init__(self) -> None:
        self._metrics_pipeline = []

    def _check_metrics(self, metric_instance: Metric):
        raise NotImplementedError("Check metrics is not implemented")
    
    def add_metric(self, metric_instance: Metric):
        self._check_metrics(metric_instance)
        self._metrics_pipeline.append(metric_instance)
        return self

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Fit is not implemented")


class BinaryEvaluator(Evaluator):

    def __init__(self):
        super(BinaryEvaluator, self).__init__()

    def _check_metrics(self, metric_instance: Metric):
        if metric_instance.__class__ not in BINARY_METRICS:
            raise ValueError("Metric {} is not a binary classification metric".format(metric_instance.name))

    def fit(self, y_true, y_pred):
        rs = {}
        for metric in self._metrics_pipeline:
            rs[metric.alias] = metric(y_true, y_pred)
        return rs
    

class RegressionEvaluator(Evaluator):

    def __init__(self):
        super(RegressionEvaluator, self).__init__()

    def _check_metrics(self, metric_instance: Metric):
        if metric_instance.__class__ not in REGRESSION_METRICS:
            raise ValueError("Metric {} is not a regression metric".format(metric_instance.name))

    def fit(self, y_true, y_pred):
        rs = {}
        for metric in self._metrics_pipeline:
            rs[metric.alias] = metric(y_true, y_pred)
        return rs


class MultiClassEvaluator(Evaluator):

    def __init__(self):
        super(MultiClassEvaluator, self).__init__()

    def _check_metrics(self, metric_instance: Metric):
        if metric_instance.__class__ not in MULTICLASS_METRICS:
            raise ValueError("Metric {} is not a multi classification metric".format(metric_instance.name))

    def fit(self, y_true, y_pred):
        rs = {}
        for metric in self._metrics_pipeline:
            rs[metric.alias] = metric(y_true, y_pred)
        return rs
