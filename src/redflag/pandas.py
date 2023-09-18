from sklearn.dummy import DummyClassifier, DummyRegressor

from .imbalance import imbalance_degree, imbalance_ratio, minority_classes
from .outliers import get_outliers
from .target import is_continuous
from .independence import is_correlated


def dummy_decorator(arg):
    """
    Returns a decorator that does nothing but wrap the function it
    decorates. Need to do this to accept an argument on the decorator.
    """
    def decorator(func):
        return func 
    return decorator


try:
    from pandas.api.extensions import register_dataframe_accessor
    from pandas.api.extensions import register_series_accessor
except:
    register_dataframe_accessor = dummy_decorator
    register_series_accessor = dummy_decorator


@register_series_accessor("redflag")
class SeriesAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def imbalance_degree(self):
        return imbalance_degree(self._obj)

    def minority_classes(self):
        return minority_classes(self._obj)
    
    def check(self):
        results = {}
        if is_continuous(self._obj):
            results['outliers'] = get_outliers(self._obj)
            results['correlated'] = is_correlated(self._obj)
        else:
            results['imbalance'] = imbalance_degree(self._obj)



@register_dataframe_accessor("redflag")
class DataFrameAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def imbalance_degree(self, target=None):
        return imbalance_degree(self._obj[target])

    def minority_classes(self, target=None):
        return minority_classes(self._obj[target])
