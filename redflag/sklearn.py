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
from sklearn.pipeline import Pipeline
from scipy.stats import wasserstein_distance
from scipy.stats import cumfreq

from .utils import is_clipped
from .target import is_continuous
from .target import is_multioutput
from .feature import is_correlated


def formatwarning(message, *args, **kwargs):
    """
    A custom warning format function.
    """
    return f"{message}\n"

class BaseRedflagDetector(BaseEstimator, TransformerMixin):

    def __init__(self, func, warning):
        self.func = func
        self.warning = warning

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Checks X (and y, if it is continuous data) for suspect values.
        """
        X = check_array(X)

        warnings.formatwarning = formatwarning

        clipped = [i for i, feature in enumerate(X.T) if self.func(feature)]
        if n := len(clipped):
            pos = ', '.join(str(c) for c in clipped)
            warnings.warn(f"🚩 Feature{'s' if n > 1 else ''} {pos} may have {self.warning} values.")

        if (y is not None) and is_continuous(y):
            if np.asarray(y).ndim == 1:
                y_ = y.reshape(-1, 1)
            for i, target in enumerate(y_.T):
                if is_clipped(target):
                    warnings.warn(f"🚩 Target {i} may have {self.warning} values.")

        return X


class ClipDetector(BaseRedflagDetector):
    """
    Transformer that detects features with clipped values.

    Example:
        >>> from sklearn.pipeline import make_pipeline
        >>> pipe = make_pipeline(ClipDetector())
        >>> X = np.array([[2, 1], [3, 2], [4, 3], [5, 3]])
        >>> pipe.fit_transform(X)  # doctest: +SKIP
        redflag/sklearn.py::redflag.sklearn.ClipDetector
          🚩 Feature 1 may have clipped values.
        array([[2, 1],
               [3, 2],
               [4, 3],
               [5, 3]])
    """
    def __init__(self):
        super().__init__(is_clipped, "clipped")


class CorrelationDetector(BaseRedflagDetector):
    """
    Transformer that detects features correlated to themselves.

    Example:
        >>> from sklearn.pipeline import make_pipeline
        >>> pipe = make_pipeline(CorrelationDetector())
        >>> rng = np.random.default_rng(0)
        >>> X = np.stack([rng.uniform(size=20), np.sin(np.linspace(0, 1, 20))]).T
        >>> pipe.fit_transform(X)  # doctest: +SKIP
        redflag/sklearn.py::redflag.sklearn.CorrelationDetector
          🚩 Feature 1 may have correlated values.
        array([[0.38077051, 0.        ],
               [0.42977406, 0.05260728]
               ...
               [0.92571458, 0.81188195],
               [0.7482485 , 0.84147098]])
    """
    def __init__(self):
        super().__init__(is_correlated, "correlated")


class DistributionComparator(BaseEstimator, TransformerMixin):
    """
    Transformer that raises warnings if validation or prediction data is not
    in the same distribution as the training data.

    Methods:
        fit_transform(): Called when fitting. In this transformer, we don't
            transform the data, we just learn the distributions.
        transform(): Called when transforming validation or prediction data.
        fit(): Called by fit_transform() when fitting the training data.
    """

    def __init__(self, threshold=1.0, bins=200, warn=True, warn_if_zero=False):
        """
        Constructor for the class.

        Args:
            threshold (float): The threshold for the Wasserstein distance.
            bins (int): The number of bins to use when computing the histograms.
            warn (bool): Whether to raise a warning or raise an error.
            warn_if_zero (bool): Whether to raise a warning if the histogram is
                identical to the training data.
        """
        self.threshold = threshold
        self.bins = bins
        self.warn = warn
        self.warn_if_zero = warn_if_zero

    def fit(self, X, y=None):
        """
        Record the histograms of the input data, using 200 bins by default.
       
        Normally we'd compute Wasserstein distance directly from the data, 
        but that seems memory-expensive.

        Args:
            X (np.ndarray): The data to learn the distributions from.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            self.
        """
        X = check_array(X)
        hists = [cumfreq(feature, numbins=self.bins) for feature in X.T]
        self.hist_counts = [h.cumcount for h in hists]
        self.hist_lowerlimits = [h.lowerlimit for h in hists]
        self.hist_binsizes = [h.binsize for h in hists]
        return self

    def transform(self, X, y=None):
        """
        Compare the histograms of the input data X to the histograms of the
        training data. We use the Wasserstein distance to compare the
        distributions.

        This transformer does not transform the data, it just compares the
        distributions and raises a warning if the Wasserstein distance is
        above the threshold.

        Args:
            X (np.ndarray): The data to compare to the training data.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            X.
        """
        X = check_array(X)
        
        # If there aren't enough samples, just return X.
        # training_examples = self.hist_counts[0][-1]
        if len(X) <  100:
            return X

        # If we have enough samples, let's carry on.
        for i, (weights, lowerlimit, binsize, feature) in enumerate(zip(self.hist_counts, self.hist_lowerlimits, self.hist_binsizes, X.T)):

            values = lowerlimit + np.linspace(0, binsize*weights.size, weights.size)
            
            hist = cumfreq(feature, numbins=self.bins)
            f_weights = hist.cumcount
            f_values = hist.lowerlimit + np.linspace(0, hist.binsize*f_weights.size, f_weights.size)

            w = wasserstein_distance(values, f_values, weights, f_weights)

            if w == 0 and self.warn_if_zero:
                warnings.formatwarning = formatwarning
                warnings.warn(f"🚩 Feature {i} is identical to the training data.")
            elif w > self.threshold:
                if self.warn:
                    warnings.formatwarning = formatwarning
                    warnings.warn(f"🚩 Feature {i} has a distribution that is different from training.")
                else:
                    raise ValueError(f"🚩 Feature {i} has a distribution that is different from training.")

        return X
    
    def fit_transform(self, X, y=None):
        """
        This is called when fitting, if it is present. We can make our call to self.fit()
        and not bother calling self.transform(), because we're not actually transforming
        anything, we're just getting set up for applying our test later during prediction.

        Args:
            X (np.ndarray): The data to compare to the training data.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            X.
        """
        # Call fit() to learn the distributions.
        self = self.fit(X, y=y)
        
        # When fitting, we do not run transform() (actually a test).
        return X


pipeline = Pipeline(
    steps=[
        ("rf.clip", ClipDetector()),
        ("rf.corr", CorrelationDetector()),
        ("rf.dist", DistributionComparator()),
    ]
)
