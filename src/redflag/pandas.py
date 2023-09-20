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

from .imbalance import imbalance_degree, imbalance_ratio, minority_classes
from .outliers import get_outliers
from .target import is_continuous, dummy_classification_scores, dummy_regression_scores
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
    'continuous': """
Outliers:    {outliers}
Correlated:  {correlated}
Dummy scores:{dummy_regression_scores}
""",
    'categorical': """
Imbalance degree: {imbalance_degree}
Minority classes: {minority_classes}
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

    def dummy_scores(self, task=None, random_state=None):
        if task is None:
            task = 'regression' if is_continuous(self._obj) else 'classification'

        if task == 'classification':
            scores = dummy_classification_scores(self._obj, random_state=random_state)
        elif task == 'regression':
            scores = dummy_regression_scores(self._obj)
        else:
            raise ValueError("`task` must be 'classification' or 'regression', or None to decide automatically.")

        return scores

    def report(self, random_state=None):
        results = {}
        if is_continuous(self._obj):
            results['outliers'] = get_outliers(self._obj)
            results['correlated'] = is_correlated(self._obj)
            results['dummy_scores'] = dummy_regression_scores(self._obj)
            template = TEMPLATES['continuous']
        else:
            # Categorical.
            min_class = minority_classes(self._obj)
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
