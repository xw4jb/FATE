from enum import Enum
from fate.ml.evaluation.metrics.regression import MAE, MedianAbsoluteError, R2Score, RMSE, MSE, ExplainedVariance
from fate.ml.evaluation.metrics.classification import BiClassAccuracy, BiClassPrecision, BiClassRecall, \
    KS, Lift, Gain, AUC, FScore, PSI
from fate.ml.evaluation.metrics.classification import MultiClassAccuracy, MultiClassPrecision, MultiClassRecall


REGRESSION_METRICS = set(
    [MAE, MedianAbsoluteError, R2Score, RMSE, MSE, ExplainedVariance]
)

BINARY_METRICS = set(
    [BiClassAccuracy, BiClassPrecision, BiClassRecall, KS, Lift, Gain, AUC, FScore, PSI]
)

MULTICLASS_METRICS = set(
    [MultiClassAccuracy, MultiClassPrecision, MultiClassRecall]
)

POSSION_REGRESSION_METRIC = set(

)

