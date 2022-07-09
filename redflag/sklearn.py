"""
Scikit-learn components.

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0

Copyright 2022 Agile Scientific

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
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy.stats import wasserstein_distance

from .utils import is_clipped
from .feature import is_correlated


def formatwarning(message, *args, **kwargs):
    return f"{message}\n"


class ClipDetector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        for i, feature in enumerate(X.T):
            if is_clipped(feature):
                warnings.formatwarning = formatwarning
                warnings.warn(f"Feature {i} may have clipped values.")
        return X


class CorrelationDetector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        for i, feature in enumerate(X.T):
            if is_correlated(feature):
                warnings.formatwarning = formatwarning
                warnings.warn(f"Feature {i} may have non-independent records.")
        return X


class DistributionComparator(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_array(X)
        self.training_data = X
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        for i, (f_train, f_this) in enumerate(zip(self.training_data.T, X.T)):
            w = wasserstein_distance(f_train, f_this)
            if w > self.threshold:
                warnings.formatwarning = formatwarning
                warnings.warn(f"Feature {i} has a distribution that is different from training.")
        return X
