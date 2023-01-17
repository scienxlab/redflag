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
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance
from scipy.stats import cumfreq
from sklearn.utils.metaestimators import available_if

from .utils import is_clipped, proportion_to_stdev, stdev_to_proportion
from .target import is_continuous
from .independence import is_correlated
from .outliers import has_outliers, expected_outliers
from .imbalance import imbalance_degree, imbalance_ratio, minority_classes
from .importance import feature_importances
from .importance import least_important_features, most_important_features


def formatwarning(message, *args, **kwargs):
    """
    A custom warning format function.
    """
    return f"{message}\n"

warnings.formatwarning = formatwarning


class BaseRedflagDetector(BaseEstimator, TransformerMixin):

    def __init__(self, func, warning, **kwargs):
        self.func = lambda X: func(X, **kwargs)
        self.warning = warning

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Checks X (and y, if it is continuous data) for suspect values.
        """
        X = check_array(X)

        positive = [i for i, feature in enumerate(X.T) if self.func(feature)]
        if n := len(positive):
            pos = ', '.join(str(i) for i in positive)
            warnings.warn(f"🚩 Feature{'s' if n > 1 else ''} {pos} may have {self.warning}.")

        if (y is not None) and is_continuous(y):
            if np.asarray(y).ndim == 1:
                y_ = y.reshape(-1, 1)
            for i, target in enumerate(y_.T):
                if self.func(target):
                    warnings.warn(f"🚩 Target {i} may have {self.warning}.")

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
        super().__init__(is_clipped, "clipped values")


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
        super().__init__(is_correlated, "correlated values")


class UnivariateOutlierDetector(BaseRedflagDetector):
    """
    Transformer that detects if there are more than the expected number of
    outliers in each feature considered separately. (To consider all features
    together, use the `OutlierDetector` instead.)

    kwargs are passed to `has_outliers`.

    Example:
        >>> from sklearn.pipeline import make_pipeline
        >>> pipe = make_pipeline(UnivariateOutlierDetector())
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(1_000, 2))
        >>> pipe.fit_transform(X)  # doctest: +SKIP
        redflag/sklearn.py::redflag.sklearn.UnivariateOutlierDetector
          🚩 Features 0, 1 may have more outliers (in a univariate sense) than expected.
        array([[ 0.12573022, -0.13210486],
               [ 0.64042265,  0.10490012],
               [-0.53566937,  0.36159505],
               ...,
               [ 1.24972527,  0.75063397],
               [-0.55581573, -2.01881162],
               [-0.90942756,  0.36922933]])
        >>> pipe = make_pipeline(UnivariateOutlierDetector(factor=2))
        >>> pipe.fit_transform(X)  # No warning.
        array([[ 0.12573022, -0.13210486],
               [ 0.64042265,  0.10490012],
               [-0.53566937,  0.36159505],
               ...,
               [ 1.24972527,  0.75063397],
               [-0.55581573, -2.01881162],
               [-0.90942756,  0.36922933]])
    """
    def __init__(self, **kwargs):
        super().__init__(has_outliers, "more outliers (in a univariate sense) than expected", **kwargs)


class MultivariateOutlierDetector(BaseEstimator, TransformerMixin):
    """
    Transformer that detects if there are more than the expected number of
    outliers when the dataset is considered as a whole, in a mutlivariate
    sense. (To consider feature distributions separately, use the
    `UnivariateOutlierDetector` instead.)

    Example:
        >>> from sklearn.pipeline import make_pipeline
        >>> pipe = make_pipeline(MultivariateOutlierDetector())
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(1_000, 2))
        >>> pipe.fit_transform(X)  # doctest: +SKIP
        redflag/sklearn.py::redflag.sklearn.MultivariateOutlierDetector
          🚩 Dataset may have more outliers (in a multivariate sense) than expected.
        array([[ 0.12573022, -0.13210486],
               [ 0.64042265,  0.10490012],
               [-0.53566937,  0.36159505],
               ...,
               [ 1.24972527,  0.75063397],
               [-0.55581573, -2.01881162],
               [-0.90942756,  0.36922933]])
        >>> pipe = make_pipeline(MultivariateOutlierDetector(factor=2))
        >>> pipe.fit_transform(X)  # No warning.
        array([[ 0.12573022, -0.13210486],
               [ 0.64042265,  0.10490012],
               [-0.53566937,  0.36159505],
               ...,
               [ 1.24972527,  0.75063397],
               [-0.55581573, -2.01881162],
               [-0.90942756,  0.36922933]])
    """
    def __init__(self, p=0.99, threshold=None, factor=1):
        self.p = p if threshold is None else None
        self.threshold = threshold
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Checks X (and y, if it is continuous data) for outlier values.
        """
        X = check_array(X)

        if X.shape[1] < 2:
            warnings.warn("MultiVariateOutlierDetector requires at least 2 features; use UnivariateOutlierDetector on this dataset.")
            return X

        outliers = has_outliers(X, p=self.p, threshold=self.threshold, factor=self.factor)

        if outliers:
            warnings.warn(f"🚩 Dataset may have more outliers (in a multivariate sense) than expected.")

        if (y is not None) and is_continuous(y):
            if np.asarray(y).ndim == 1:
                y_ = y.reshape(-1, 1)
            if has_outliers(y_, p=self.p, threshold=self.threshold, factor=self.factor):
                    warnings.warn(f"🚩 Target may have more outliers (in a multivariate sense) than expected.")

        return X


class DistributionComparator(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=1.0, bins=200, warn=True, warn_if_zero=False):
        """
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

        Sets `self.histograms` to the learned histograms.

        Args:
            X (np.ndarray): The data to learn the distributions from.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            self.
        """
        X = check_array(X)
        self.histograms_ = [cumfreq(feature, numbins=self.bins) for feature in X.T]
        self.hist_counts = [h.cumcount for h in self.histograms_]
        self.hist_lowerlimits = [h.lowerlimit for h in self.histograms_]
        self.hist_binsizes = [h.binsize for h in self.histograms_]
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
        wasserstein_distances = []
        for i, (weights, lowerlimit, binsize, feature) in enumerate(zip(self.hist_counts, self.hist_lowerlimits, self.hist_binsizes, X.T)):

            values = lowerlimit + np.linspace(0, binsize*weights.size, weights.size)
            
            hist = cumfreq(feature, numbins=self.bins)
            f_weights = hist.cumcount
            f_values = hist.lowerlimit + np.linspace(0, hist.binsize*f_weights.size, f_weights.size)

            w = wasserstein_distance(values, f_values, weights, f_weights)
            wasserstein_distances.append(w)

        W = np.array(wasserstein_distances)

        zeros = np.where(W == 0)[0]
        if n := zeros.size and self.warn_if_zero:
            warnings.warn(f"🚩 Feature{'s' if n > 1 else ''} {pos} {'are' if n > 1 else 'is'} identical to the training data.")

        positive = np.where(W > self.threshold)[0]
        if n := positive.size:
            pos = ', '.join(str(i) for i in positive)
            if self.warn:
                warnings.warn(f"🚩 Feature{'s' if n > 1 else ''} {pos} {'have distributions that are' if n > 1 else 'has a distribution that is'} different from training.")
            else:
                raise ValueError(f"🚩 Feature{'s' if n > 1 else ''} {pos} {'have distributions that are' if n > 1 else 'has a distribution that is'} different from training.")

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


class OutlierDetector(BaseEstimator, TransformerMixin):

    def __init__(self, p=0.99, threshold=None, factor=1.0):
        """
        Constructor for the class.

        Args:
            p (float): The confidence level.
            threshold (float): The threshold for the Wasserstein distance.
        """
        self.threshold = threshold
        self.p = p if threshold is None else None
        self.factor = factor

    def _actual_vs_expected(self, z, n, d):
        """
        Calculate the expected number of outliers in the data.
        """
        # Calculate the Mahalanobis distance threshold if necessary.
        if self.threshold is None:
            self.threshold = proportion_to_stdev(p=self.p, d=d)
        else:
            self.p = stdev_to_proportion(self.threshold, d=d)

        # Decide whether each point is an outlier or not.
        idx, = np.where((z < -self.threshold) | (z > self.threshold))

        # Calculate the expected number of outliers in the training data.
        expected = int(self.factor * expected_outliers(n, d, threshold=self.threshold))

        # If the number of outliers is greater than the expected number, raise a warning.
        return idx, expected

    def fit(self, X, y=None):
        """
        Record the robust location and covariance.

        Sets `self.outliers_` to the indices of the outliers beyond the given
        threshold distance.

        Args:
            X (np.ndarray): The data to learn the distributions from.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            self.
        """
        X = check_array(X)
        n, d = X.shape

        # Fit the distributions.
        self.ee = EllipticEnvelope(support_fraction=1.0).fit(X)

        # Compute the Mahalanobis distance of the training data.
        z = np.sqrt(self.ee.dist_)

        self.outliers_, expected = self._actual_vs_expected(z, n, d)
        if self.outliers_.size > expected:
            warnings.warn(f"🚩 There are more outliers than expected in the training data ({self.outliers_.size} vs {expected}).")

        return self

    def transform(self, X, y=None):
        """
        Compute the Mahalanobis distances using the location and covarianced
        learned from the training data.

        This transformer does not transform the data, it just compares the
        distributions and raises a warning if there are more outliers than
        expected, given the confidence level or threshold specified at
        instantiation.

        Args:
            X (np.ndarray): The data to compare to the training data.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            X.
        """
        X = check_array(X)
        n, d = X.shape

        # Compute the Mahalanobis distances for the given data, using the
        # learned location and covariance.
        z = np.sqrt(self.ee.mahalanobis(X))

        actual, expected = self._actual_vs_expected(z, n, d)
        if actual.size > expected:
            warnings.warn(f"🚩 There are more outliers than expected in the data ({actual.size} vs {expected}).")

        return X

    def fit_transform(self, X, y=None):
        """
        This is called when fitting, if it is present. We can make our call to self.fit()
        and not bother calling self.transform(), because we're not actually transforming
        anything, we're just getting set up for applying our test later during prediction.
        The warning about outliers in the data will come from self.fit().

        Args:
            X (np.ndarray): The data to compare to the training data.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            X.
        """
        self = self.fit(X, y=y)
        
        # When fitting, we do not run transform().
        return X


class ImbalanceDetector(BaseEstimator, TransformerMixin):

    def __init__(self, method='id', threshold=0.4, classes=None):
        """
        Constructor for the class.

        Args:
            method (str): The method to use for imbalance detection. In general,
                'id' is the best method for multi-class classification problems
                (but can be used for binary classification problems as well).
            threshold (float): The threshold for the imbalance, default 0.5.
                For 'id', the imbalance summary statistic is in [0, 1). See
                Ortigosa-Hernandez et al. (2017) for details. For 'ir', the
                threshold is a ratio of the majority class to the minority class
                and ranges from 1 (balanced) to infinity (nothing in the
                minority class).
            classes (list): The names of the classes present in the data, even
                if they are not present in the array `y`.
        """
        if method not in ['id', 'ir']:
            raise ValueError(f"Method must be 'id' or 'ir' but was {method}")

        if (method == 'ir') and (threshold <= 1):
            raise ValueError(f"Method is 'ir' but threshold <= 1. For IR, the measure is the ratio of the majority class to the minority class; for example use 2 to trigger a warning if there are twice as many samples in the majority class as in the minority class.")

        if (method == 'id') and (threshold >= 1):
            raise ValueError(f"Method is 'id' but threshold >= 1. For ID, the measure is always in [0, 1).")

        self.method = method
        self.threshold = threshold
        self.classes = classes

    def fit(self, X, y=None):
        """
        Checks y for imbalance.

        Sets `self.minority_classes_` and `self.imbalance_`. Note: imbalance
        degree is adjusted to express only the fractional part; for the integer
        part, use the length of the minority class list).

        Args:
            X (np.ndarray): The data to compare to the training data. Not used
                by this transformer.
            y (np.ndarray): The labels for the data.

        Returns:
            self.
        """
        # If there's no target or y is continuous (probably a regression), we're done.
        if y is None or is_continuous(y):
            warnings.warn("Target y is None or seems continuous, so no imbalance detection.")
            return self

        methods = {'id': imbalance_degree, 'ir': imbalance_ratio}
        imbalance = methods[self.method](y)
        
        if self.method == 'id':
            imbalance = imbalance - int(imbalance)

        self.imbalance_ = imbalance
        self.minority_classes_ = minority_classes(y, classes=self.classes)

        imbalanced = (len(self.minority_classes_) > 0) and (imbalance > self.threshold)

        if imbalanced and self.method == 'id':
            warnings.warn(f"🚩 The labels are imbalanced by more than the threshold ({imbalance:0.3f} > {self.threshold:0.3f}). See self.minority_classes_ for the minority classes.")
        if imbalanced and self.method == 'ir':
            warnings.warn(f"🚩 The labels are imbalanced by more than the threshold ({imbalance:0.1f} > {self.threshold:0.1f}). See self.minority_classes_ for the minority classes.")

        return self

    def transform(self, X, y=None):
        """
        This detector does nothing during 'transform', only during 'fit'.

        Args:
            X (np.ndarray): The data to compare to the training data. Not used
                by this transformer.
            y (np.ndarray): The labels for the data.

        Returns:
            X.
        """
        return check_array(X)


class ImbalanceComparator(BaseEstimator, TransformerMixin):

    def __init__(self, method='id', threshold=0.4, min_class_diff=1, classes=None):
        """
        Args:
            method (str): The method to use for imbalance detection. In general,
                'id' is the best method for multi-class classification problems
                (but can be used for binary classification problems as well).
            threshold (float): The threshold for the imbalance, default 0.5.
                For 'id', the imbalance summary statistic is in [0, 1). See
                Ortigosa-Hernandez et al. (2017) for details. For 'ir', the
                threshold is a ratio of the majority class to the minority class
                and ranges from 1 (balanced) to infinity (nothing in the
                minority class).
            min_class_diff (int): The difference in the number of minority
                classes that will trigger a warning.
            classes (list): The names of the classes present in the data, even
                if they are not present in the array `y`.
        """
        if method not in ['id', 'ir']:
            raise ValueError(f"Method must be 'id' or 'ir' but was {method}")

        if (method == 'ir') and (threshold <= 1):
            raise ValueError(f"Method is 'ir' but threshold <= 1. For IR, the measure is the ratio of the majority class to the minority class; for example use 2 to trigger a warning if there are twice as many samples in the majority class as in the minority class.")

        if (method == 'id') and (threshold >= 1):
            raise ValueError(f"Method is 'id' but threshold >= 1. For ID, the measure is always in [0, 1).")

        self.method = method
        self.threshold = threshold
        self.min_class_diff = min_class_diff
        self.classes = classes

    def fit(self, X, y=None):
        """
        Record the imbalance degree and minority classes of the input data.

        Sets `self.minority_classes_` and `self.imbalance_`.

        Args:
            X (np.ndarray): The data to learn the statistics from.
            y (np.ndarray): The labels for the data. Not used for anything.

        Returns:
            self.
        """
        # If there's no target or y is continuous (probably a regression), we're done.
        if y is None or is_continuous(y):
            warnings.warn("Target y is None or seems continuous, so no imbalance detection.")
            return self

        methods = {'id': imbalance_degree, 'ir': imbalance_ratio}
        imbalance = methods[self.method](y)
        
        if self.method == 'id':
            imbalance = imbalance - int(imbalance)

        self.imbalance_ = imbalance
        self.minority_classes_ = minority_classes(y, classes=self.classes)
        return self

    def transform(self, X, y=None):
        """
        Compare the imbalance statistics of the labels, y, between the
        training data (calling `fit`) and subsequent data (calling `transform`).

        This transformer does not transform the data, it just compares the
        distributions.

        Args:
            X (np.ndarray): The data to compare to the training data. Not used.
            y (np.ndarray): The labels for the data.

        Returns:
            X.
        """
        # If there's no target or y is continuous (probably a regression), we're done.
        if y is None or is_continuous(y):
            warnings.warn("Target y is None or seems continuous, so no imbalance detection.")
            return self

        methods = {'id': imbalance_degree, 'ir': imbalance_ratio}
        imbalance = methods[self.method](y)

        if self.method == 'id':
            imbalance = imbalance - int(imbalance)

        min_classes = minority_classes(y, classes=self.classes)

        diff = abs(len(min_classes) - len(self.minority_classes_))

        # Check if there's a different *number* of minority classes.
        if diff >= self.min_class_diff:
            warnings.warn(f"🚩 There is a different number of minority classes ({len(min_classes)}) compared to the training data ({len(self.minority_classes_)}).")

        # Check if there's the same number but the minority classes have changed.
        if set(min_classes) != set(self.minority_classes_):
            warnings.warn(f"🚩 The minority classes ({', '.join(str(c) for c in set(min_classes))}) are different from those in the training data ({', '.join(str(c) for c in set(self.minority_classes_))}).")

        # Check if the imbalance metric has changed.
        if abs(imbalance - self.imbalance_) >= self.threshold:
            warnings.warn(f"🚩 The imbalance metric ({imbalance}) is different from that of the training data ({self.imbalance_}).")

        return check_array(X)

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
        
        # When fitting, we do not run transform().
        return check_array(X)


class ImportanceDetector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=None, random_state=None):
        """
        Constructor for the class.

        Args:
            threshold (float): The threshold for the cumulative importance.
            max_threshold (float): The maximum threshold for the importance
                of a single feature.
        """
        if (threshold is not None) and not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be between 0 and 1, but was {threshold}")

        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Checks the dataset (X and y together) for unusually low and/or high
            importance.

        Args:
            X (np.ndarray): The data. Not used by this detector.
            y (np.ndarray): The labels for the data.

        Returns:
            X.
        """
        if y is None:
            warnings.warn("Target y is None, so no importance detection.")
            return self

        importances = feature_importances(X, y, random_state=self.random_state)
        most_important = most_important_features(importances, threshold=self.threshold)

        if (m := len(most_important)) <= 2:
            most_str = ', '.join(str(i) for i in most_important)
            warnings.warn(f"🚩 Feature{'' if m == 1 else 's'} {most_str} {'has' if m == 1 else 'have'} very high importance; check for leakage.")
            return self

        # Don't do this check if there were high-importance features (infer that the others are low.)
        least_important = least_important_features(importances, threshold=self.threshold)
        if (m := len(least_important)) > 0:
            least_str = ', '.join(str(i) for i in least_important)
            warnings.warn(f"🚩 Feature{'' if m == 1 else 's'} {least_str} {'has' if m == 1 else 'have'} low importance; check for relevance.")

        return self

    def transform(self, X, y=None):
        """
        This detector does nothing during 'transform', only during 'fit'.

        Args:
            X (np.ndarray): The data. Not used by this detector.
            y (np.ndarray): The labels for the data.

        Returns:
            X.
        """
        return check_array(X)


class RfPipeline(pipeline.Pipeline):

    """
    This class is adapted from original Pipeline code at sklearn/pipeline.py
    (c) the scikit-learn contributors and licensed under BSD 3-clause license.
    """

    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
            )

    @available_if(_can_transform)
    def transform(self, X, y=None):
        """
        Required because built-in sklearn pipeline does not handle y.

        Transform the data, and apply `transform` with the final estimator.
        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.
        
        This also works where final estimator is `None` in which case all prior
        transformations are applied.
        
        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable
            Target vector. Optional.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt = X
        for _, _, transform in self._iter():
            if y is None:
                Xt = transform.transform(Xt)
            else:
                Xt = transform.transform(Xt, y)
        return Xt


def make_rf_pipeline(*steps, memory=None, verbose=False):
    """Construct a :class:`RfPipeline` from the given estimators.
    This is a shorthand for the :class:`RfPipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    This function is adapted from original code at sklearn/pipeline.py
    (c) the scikit-learn contributors and licensed under BSD 3-clause license.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : RfPipeline
        Returns a :class:`RfPipeline` object.
    """
    return RfPipeline(_name_estimators(steps), memory=memory, verbose=verbose)


pipeline = Pipeline(
    steps=[
        ("rf.imbalance", ImbalanceDetector()),
        ("rf.clip", ClipDetector()),
        ("rf.correlation", CorrelationDetector()),
        ("rf.outlier", OutlierDetector()),
        ("rf.distributions", DistributionComparator()),
        ("rf.importance", ImportanceDetector()),
    ]
)
