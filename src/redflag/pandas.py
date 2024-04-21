"""
Pandas accessors.

Author: Matt Hall, scienxlab.org
Licence: Apache 2.0

Copyright 2024 Redflag contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
from typing import Optional

from .imbalance import imbalance_degree, minority_classes, is_imbalanced
from .importance import feature_importances as feature_importances
from .outliers import get_outliers
from .target import *
from .independence import is_correlated
from .utils import docstring_from


def null_decorator(arg):
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
    register_dataframe_accessor = null_decorator
    register_series_accessor = null_decorator


TEMPLATES = {
    'continuous': """Continuous data suitable for regression
Outliers:    {outliers}
Correlated:  {correlated}
Dummy scores:{dummy_scores}
""",
    'categorical': """Categorical data suitable for classification
Imbalance degree: {imbalance}
Minority classes: {minority_classes}
Dummy scores:     {dummy_scores}
"""
}

@register_series_accessor("redflag")
class SeriesAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @docstring_from(minority_classes)
    def minority_classes(self):
        if is_continuous(self._obj):
            warnings.warn('The Series does not seem categorical.')
        return minority_classes(self._obj)

    @docstring_from(imbalance_degree)
    def imbalance_degree(self):
        if is_continuous(self._obj):
            warnings.warn('The Series does not seem categorical.')
        return imbalance_degree(self._obj)

    @docstring_from(is_imbalanced)
    def is_imbalanced(self, threshold=0.4, method='tv', classes=None):
        if is_continuous(self._obj):
            warnings.warn('The Series does not seem categorical.')
        return is_imbalanced(self._obj,
                             threshold=threshold,
                             method=method,
                             classes=classes
                             )

    @docstring_from(is_ordered)
    def is_ordered(self, q=0.95):
        return is_ordered(self._obj, q=q)

    @docstring_from(dummy_scores)
    def dummy_scores(self, task='auto', random_state=None):
        return dummy_scores(self._obj, task=task, random_state=random_state)

    def report(self, random_state=None):
        results = {}
        if is_continuous(self._obj):
            results['outliers'] = get_outliers(self._obj)
            results['correlated'] = is_correlated(self._obj)
            results['dummy_scores'] = dummy_regression_scores(self._obj)
            template = TEMPLATES['continuous']
        else:
            # Categorical.
            results['minority_classes'] = minority_classes(self._obj)
            results['imbalance'] = imbalance_degree(self._obj)
            results['dummy_scores'] = dummy_classification_scores(self._obj, random_state=random_state)
            template = TEMPLATES['categorical']

        return template.format(**results)


@register_dataframe_accessor("redflag")
class DataFrameAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @docstring_from(feature_importances)
    def feature_importances(self, features=None, target=None,
                            n: int=3, task: Optional[str]=None,
                            random_state: Optional[int]=None,
                            standardize: bool=True):
        if target is None:
            raise ValueError('You must provide a target column.')
        else:
            y_ = self._obj[target]
            if is_continuous(y_):
                task = 'regression'
            else:
                task = 'classification'
        if len(y_.shape) > 1:
            raise NotImplementedError('Multilabel targets are not supported.')
        if features is None and target is not None:
            X_ = self._obj.drop(columns=target)
        else:
            X_ = self._obj[features]
        return feature_importances(X_, y_, n=n, task=task,
                                   random_state=random_state,
                                   standardize=standardize)


    def correlation_detector(self, features=None, target=None, n=20, s=20, threshold=0.1):
        """
        This is an experimental feature.
        """
        if target is not None:
            y_ = self._obj[target]
            if len(y_.shape) > 1:
                raise NotImplementedError('Multilabel targets are not supported.')
            if is_correlated(y_):
                warnings.warn('The target appears to be autocorrelated.',stacklevel=2)

        if features is None and target is not None:
            X_ = self._obj.drop(target, axis=1).values
        else:
            X_ = self._obj[features].values

        for i, x in enumerate(X_.T):
            if is_correlated(x, n=n, s=s, threshold=threshold):
                warnings.warn(f'🚩 Feature {i} appears to be autocorrelated.', stacklevel=2)

        # There is probably something more useful to return.
        return
