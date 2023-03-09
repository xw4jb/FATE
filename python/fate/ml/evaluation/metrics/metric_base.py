from enum import Enum

class MetricType(Enum):
    BINARY = 'binary'
    MULTI = 'multi'
    REGRESSION = 'regression'


class Metric(object):

    name = None
    metric_type = None
    alias = None

    def __init__(self) -> None:
        self._result = None

    def fit(self, *args, **kwargs):
        pass

    def get_callback_result(self):
        pass

    def __call__(self, *args, **kwds):
        return self.fit(*args, **kwds)
