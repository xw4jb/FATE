from scipy.stats import stats
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from fate.ml.evaluation.metrics.metric_base import Metric, MetricType
import numpy as np


class RMSE(Metric):
    name = 'RootMeanSquaredError'
    metric_type = MetricType.REGRESSION
    alias = 'rmse'

    def __init__(self):
        super().__init__()

    def fit(self, labels, pred_scores):
        self._result = np.sqrt(mean_squared_error(labels, pred_scores))
        return self._result


class MAE(Metric):
    name = 'MeanAbsoluteError'
    metric_type = MetricType.REGRESSION
    alias = 'mae'

    def __init__(self):
        super().__init__()

    def fit(self, labels, pred_scores):
        self._result = mean_absolute_error(labels, pred_scores)
        return self._result


class R2Score(Metric):
    name = 'R2Score'
    metric_type = MetricType.REGRESSION
    alias = 'r2'

    def __init__(self):
        super().__init__()

    def fit(self, labels, pred_scores):
        self._result = r2_score(labels, pred_scores)
        return self._result


class MSE(Metric):
    name = 'MeanSquaredError'
    metric_type = MetricType.REGRESSION
    alias = 'mse'

    def __init__(self):
        super().__init__()

    def fit(self, labels, pred_scores):
        self._result = mean_squared_error(labels, pred_scores)
        return self._result


class ExplainedVariance(Metric):
    name = 'ExplainedVariance'
    metric_type = MetricType.REGRESSION
    alias = 'ev'

    def __init__(self):
        super().__init__()

    def fit(self, labels, pred_scores):
        self._result = explained_variance_score(labels, pred_scores)
        return self._result


class MedianAbsoluteError(Metric):
    name = 'MedianAbsoluteError'
    metric_type = MetricType.REGRESSION
    alias = 'medae'

    def __init__(self):
        super().__init__()

    def fit(self, labels, pred_scores):
        self._result = median_absolute_error(labels, pred_scores)
        return self._result