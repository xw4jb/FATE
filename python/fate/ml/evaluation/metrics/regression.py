from scipy.stats import stats
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from fate.ml.evaluation.metrics.metric_base import Metric, MetricType
import numpy as np


class RMSE(Metric):

    def __init__(self):
        super().__init__('RootMeanSquaredError', MetricType.REGRESSION, 'rmse')

    def fit(self, labels, pred_scores):
        self._result = np.sqrt(mean_squared_error(labels, pred_scores))
        return self._result


class MAE(Metric):

    def __init__(self):
        super().__init__('MeanAbsoluteError', MetricType.REGRESSION, 'mae')

    def fit(self, labels, pred_scores):
        self._result = mean_absolute_error(labels, pred_scores)
        return self._result


class R2Score(Metric):

    def __init__(self):
        super().__init__('R2Score', MetricType.REGRESSION, 'r2')

    def fit(self, labels, pred_scores):
        self._result = r2_score(labels, pred_scores)
        return self._result


class MSE(Metric):

    def __init__(self):
        super().__init__('MeanSquaredError', MetricType.REGRESSION, 'mse')

    def fit(self, labels, pred_scores):
        self._result = mean_squared_error(labels, pred_scores)
        return self._result


class ExplainedVariance(Metric):

    def __init__(self):
        super().__init__('ExplainedVariance', MetricType.REGRESSION, 'ev')

    def fit(self, labels, pred_scores):
        self._result = explained_variance_score(labels, pred_scores)
        return self._result


class MedianAbsoluteError(Metric):

    def __init__(self):
        super().__init__('MedianAbsoluteError', MetricType.REGRESSION, 'medae')

    def fit(self, labels, pred_scores):
        self._result = median_absolute_error(labels, pred_scores)
        return self._result


class IC(Metric):

    def __init__(self):
        super().__init__('InformationCriterion', 'model selection', 'ic')

    def fit(self, k, n, dfe, loss):
        aic_score = k * dfe + 2 * n * loss
        self._result = aic_score
        return self._result


class IC_Approx(Metric):

    def __init__(self):
        super().__init__('InformationCriterionApprox', 'model selection', 'ic_approx')

    def fit(self, k, n, dfe, loss):
        aic_score = k * dfe + n * np.log(loss * 2)
        self._result = aic_score
        return self._result


class Describe(Metric):

    def __init__(self):
        super().__init__('DescriptiveStatistics', 'descriptive', 'describe')

    def fit(self, pred_scores):
        describe = stats.describe(pred_scores)
        self._result = {"min": describe.minmax[0], "max": describe.minmax[1], "mean": describe.mean,
                        "variance": describe.variance, "skewness": describe.skewness, "kurtosis": describe.kurtosis}
        return self._result
