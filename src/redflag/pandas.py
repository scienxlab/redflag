"""
Pandas accessors.

Author: Matt Hall, scienxlab.org
Licence: Apache 2.0

Copyright 2023 Redflag contributors

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

from .imbalance import imbalance_degree,minority_classes
from .outliers import get_outliers
from .target import is_continuous, dummy_scores, dummy_regression_scores, dummy_classification_scores
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

    def imbalance_degree(self):
        if is_continuous(self._obj):
            warnings.warn('The Series does not seem categorical.')
        return imbalance_degree(self._obj)

    def minority_classes(self):
        if is_continuous(self._obj):
            warnings.warn('The Series does not seem categorical.')
        return minority_classes(self._obj)

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

    # Probably should not implement these target-only methods.
    def imbalance_degree(self, target=None):
        return self._obj[target].redflag.imbalance_degree()

    def minority_classes(self, target=None):
        return self._obj[target].redflag.minority_classes()

    def dummy_scores(self, target=None, task=None, random_state=None):
        return self._obj[target].redflag.dummy_scores(task=task, random_state=random_state)
