"""
Functions related to understanding features.

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
from __future__ import annotations

from typing import Optional
from functools import reduce, partial
import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from .utils import stdev_to_proportion, proportion_to_stdev
from .utils import get_idx


def mahalanobis(X: ArrayLike, correction: bool=False) -> np.ndarray:
    """
    Compute the Mahalanobis distances of every record (row) in a 2D dataset.

    If X has a single feature, this is equivalent to computing the Z-scores
    of the data. For more features, the Mahalanobis distance is the distance
    of each point from the centroid of the data, in units analogous to the
    standard deviation. It is a multivariate analog of the Z-score.

    The empirical covariance correction factor suggested by Rousseeuw and
    Van Driessen may be optionally applied by setting `correction=True`.

    Args:
        X (array): The data. Must be a 2D array, shape (n_samples, n_features).
        correction (bool): Whether to apply the empirical covariance correction.

    Returns:
        array: The Mahalanobis distances.

    Examples:
        >>> data = np.array([-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]).reshape(-1, 1)
        >>> mahalanobis(data)
        array([1.6583124, 1.1055416, 1.1055416, 0.5527708, 0.       , 0.       ,
               0.       , 0.5527708, 1.1055416, 1.1055416, 1.6583124])
        >>> mahalanobis(data, correction=True)
        array([1.01173463, 0.67448975, 0.67448975, 0.33724488, 0.        ,
               0.        , 0.        , 0.33724488, 0.67448975, 0.67448975,
               1.01173463])
    """
    X = np.asarray(X)

    ee = EllipticEnvelope(support_fraction=1.0).fit(X)

    if correction:
        ee.correct_covariance(X)

    return np.sqrt(ee.dist_)


def mahalanobis_outliers(X: ArrayLike,
                         p: float=0.99,
                         threshold: Optional[float]=None,
                         ) -> np.ndarray:
    """
    Find outliers given samples and a threshold in multiples of stdev.
    Returns -1 for outliers and 1 for inliers (to match the sklearn API).

    For univariate data, we expect this many points outside (in units of
    standard deviation, and with equivalent p-values):
        - 1 sd: expect 31.7 points in 100 (p = 1 - 0.317 = 0.683)
        - 2 sd: 4.55 in 100 (p = 1 - 0.0455 = 0.9545)
        - 3 sd: 2.70 in 1000 (p = 1 - 0.0027 = 0.9973)
        - 4 sd: 6.3 in 100,000 (p = 1 - 0.000063 = 0.999937)
        - 4.89163847 sd: 1 in 1 million (p = 1 - 0.000001 = 0.999999)
        - 5 sd: 5.7 in 10 million datapoints
        - 6 sd: 2.0 in 1 billion points

    Args:
        X (array): The data. Can be a 2D array, shape (n_samples, n_features),
            or a 1D array, shape (n_samples).
        p (float): The probability threshold, in the range [0, 1]. This value
            is ignored if `threshold` is not None; in this case, `p` will be
            computed using `utils.stdev_to_proportion(threshold)`.
        threshold (float): The threshold in Mahalanobis distance, analogous to
            multiples of standard deviation for a single variable. If not None,
            the threshold will be used to compute `p`.

    Returns:
        array: Array identifying outliers; -1 for outliers and 1 for inliers.

    Examples:
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> mahalanobis_outliers(data)
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> mahalanobis_outliers(data + [100], threshold=3)
        array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    _, d = X.shape
        
    # Determine the Mahalanobis distance for the given confidence level.
    if threshold is None:
        threshold = proportion_to_stdev(p=p, d=d)

    # Compute the Mahalanobis distance.
    z = mahalanobis(X)

    # Decide whether each point is an outlier or not.
    idx, = np.where((z < -threshold) | (z > threshold))
    outliers = np.full(z.shape, 1)
    outliers[idx] = -1

    return outliers


def get_outliers(a: ArrayLike,
                 method: Optional[str]=None, # Can change to 'mah' in 0.6.0.
                 p: float=0.99,
                 threshold: Optional[float]=None,
                 ) -> np.ndarray:
    """
    Returns outliers in the data, considering all of the features. What counts
    as an outlier is determined by the threshold, which is in multiples of
    the standard deviation. (The conversion to 'contamination' is approximate.)

    Methods: 'iso' (isolation forest), 'lof' (local outlier factor),
    'ee' (elliptic envelope), or 'mah' (Mahanalobis distance, the default), or
    pass a function that returns an array of outlier flags (-1 for outliers and 1
    for inliers, matching the `sklearn` convention). You can also pass 'any',
    which will try all three outlier detection methods and return the outliers
    which are detected by any of them, or 'all', which will return the outliers
    which are common to all four methods. That is, 'all' is a rather conservative
    outlier detector, 'any' is rather liberal, and both of these are slower
    than choosing a single algorithm.

    Args:
        a (array): The data.
        method (str): The method to use. Can be 'mah' (the default), 'iso', 'lof',
            'ee', 'any', 'all', or a function that returns a Boolean array of
            outlier flags.
        p (float): The probability threshold, in the range [0, 1]. This value
            is ignored if `threshold` is not None; in this case, `p` will be
            computed using `utils.stdev_to_proportion(threshold)`.
        threshold (float): The threshold in Mahalanobis distance, analogous to
            multiples of standard deviation for a single variable. If not None,
            the threshold will be used to compute `p`.

    Returns:
        array: The indices of the outliers.

    Examples:
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> get_outliers(3 * data)
        array([], dtype=int64)
        >>> get_outliers(3 * data + [100])
        array([33])
        >>> get_outliers(3 * data + [100], method='mah')
        array([33])
        >>> get_outliers(3 * data + [100], method='any')
        array([33])
        >>> get_outliers(3 * data + [100], method='all')
        array([33])
    """
    if method is None:
        # Was called with the default method, which changed in 0.4.3
        method = 'mah'
        warnings.warn('The default method for get_outliers has changed to "mah". '
                      'Please specify the method explicitly to avoid this warning.',
                      DeprecationWarning, stacklevel=2)
    if p >= 1 or p < 0:
        raise ValueError('p must be in the range [0, 1).')
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if threshold is None:
        expect = 1 - p
    else:
        expect = 1 - stdev_to_proportion(threshold)
        p = 1 - expect
    methods = {
        'iso': IsolationForest(contamination=expect).fit_predict,
        'lof': LocalOutlierFactor(contamination=expect, novelty=False).fit_predict,
        'ee': EllipticEnvelope(contamination=expect).fit_predict,
        'mah': partial(mahalanobis_outliers, p=p, threshold=threshold),
    }
    if method == 'any':
        results = [get_idx(func(a)==-1) for func in methods.values()]
        outliers = reduce(np.union1d, results)
    elif method == 'all':
        results = [get_idx(func(a)==-1) for func in methods.values()]
        outliers = reduce(np.intersect1d, results)
    else:
        func = methods.get(method, method)
        outliers, = np.where(func(a)==-1)
    return outliers


def expected_outliers(n: int,
                      d: int=1,
                      p: float=0.99,
                      threshold: Optional[float]=None,
                      ) -> int:
    """
    Expected number of outliers in a dataset, under the assumption that the
    data are multivariate-normally distributed. What counts as an outlier is
    determined by the threshold, which is in multiples of the standard
    deviation, or by the p-value, which is the probability of a point being
    an outlier. Note that passing p = 0.99 does not necessarily mean that
    1% of the points will be outliers, only that 1% of the points are expected
    to be outliers, on average, if the data are normally distributed.
    
    Args:
        n (int): The number of samples.
        d (int): The number of features. Note that if threshold is None, this
            value is not used in the calculation. Default: 1.
        p (float): The probability threshold, in the range [0, 1]. This value
            is ignored if `threshold` is not None and `p` will be computed
            using `utils.stdev_to_proportion(threshold)`. Default: 0.99.
        threshold (float): The threshold in Mahalanobis distance, analogous to
            multiples of standard deviation for a single variable. If not None,
            the threshold will be used to compute `p`.
            
    Returns:
        int: The expected number of outliers.

    Example:
        >>> expected_outliers(10_000, 6, threshold=4)
        137
    """
    if threshold is not None:
        p = stdev_to_proportion(threshold, d)
    return int(n * (1 - p))


def has_outliers(a: ArrayLike,
                 p: float=0.99,
                 threshold: Optional[float]=None,
                 factor: float=1.0,
                 ) -> bool:
    """
    Use Mahalanobis distance to determine if there are more outliers than
    expected at the given confidence level or Mahalanobis distance threshold.
    A Boolean wrapper around `expected_outliers` and `get_outliers`.

    Args:
        a (array): The data. If 2D, the rows are samples and the columns are
            features. If 1D, the data are assumed to be univariate.
        p (float): The probability threshold, in the range [0, 1]. This value
            is ignored if `threshold` is not None and `p` will be computed
            using `utils.stdev_to_proportion(threshold)`. Default: 0.99.
        threshold (float): The threshold in Mahalanobis distance, analogous to
            multiples of standard deviation for a single variable. If not None,
            the threshold will be used to compute `p`.
        factor (float): The factor by which to multiply the expected number of
            outliers before comparing to the actual number of outliers.

    Returns:
        bool: True if there are more outliers than expected at the given
            confidence level.
    """
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    n, d = a.shape

    if threshold is not None:
        p = stdev_to_proportion(threshold, d)

    expected = expected_outliers(n, d, p=p)

    return get_outliers(a, method='mah', p=p).size > factor * expected
