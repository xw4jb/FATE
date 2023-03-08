from enum import Enum

class MetricType(Enum):
    BINARY = 'binary'
    MULTI = 'multi'
    REGRESSION = 'regression'


class Metric(object):

    def __init__(self, name, metric_type, alias) -> None:
        self._name = name
        self._metric_type = metric_type
        self._alias = alias
        self._result = None

    def fit(self, *args, **kwargs):
        pass

    def get_callback_result(self):
        pass

    def __call__(self, *args, **kwds):
        return self.fit(*args, **kwds)
