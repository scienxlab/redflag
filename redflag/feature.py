"""
Functions related to understanding features.

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
from collections import namedtuple
from functools import reduce
from itertools import combinations
import warnings

import numpy as np
import scipy.stats as ss
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from .utils import is_standardized, stdev_to_proportion
from .utils import get_idx
from .utils import iter_groups


DISTS = [
    'norm',
    'cosine',
    'expon',
    'exponpow',
    'gamma',
    'gumbel_l',
    'gumbel_r',
    'powerlaw',
    'triang',
    'trapz',
    'uniform',
]

def best_distribution(a, bins=None):
    """
    Model data by finding best fit distribution to data.

    Returns the best fit distribution and its parameters.

    Args:
        a (array): The data.
        bins (int): The number of bins to use for the histogram.

    Returns:
        tuple: The best fit distribution and its parameters.

    Examples:
        >>> a = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8]
        >>> best_distribution(a)
        Distribution(name='norm', shape=[], loc=4.0, scale=1.8771812708978117)
        >>> best_distribution([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7])
        Distribution(name='triang', shape=[0.5001419889107208], loc=0.3286356643172673, scale=7.3406453953773365)
    """
    if bins is None:
        bins = min(max(20, len(a) // 100), 200)
    n, x = np.histogram(a, bins=bins, density=True)
    x = (x[1:] + x[:-1]) / 2

    dists = [getattr(ss, d) for d in DISTS]

    best_dist = None
    best_params = None
    best_sse = np.inf

    for dist in dists:
        *shape, μ, σ = dist.fit(a)
        n_pred = dist.pdf(x, loc=μ, scale=σ, *shape)
        sse = np.sum((n - n_pred)**2)
        if 0 < sse < best_sse:
            best_dist = dist
            best_params = shape + [μ] + [σ]
            best_sse = sse

    *shape, μ, σ = best_params
    Distribution = namedtuple('Distribution', ['name', 'shape', 'loc', 'scale'])
    return Distribution(best_dist.name, shape, μ, σ)


def is_correlated(a, n=20, s=20, threshold=0.1):
    """
    Check if a dataset is correlated. Uses s chunks of n samples.

    Args:
        a (array): The data.
        n (int): The number of samples per chunk.
        s (int): The number of chunks.
        threshold (float): The auto-correlation threshold.

    Returns:
        bool: True if the data are correlated.

    Examples:
        >>> is_correlated([7, 1, 6, 8, 7, 6, 2, 9, 4, 2])
        False
        >>> is_correlated([1, 2, 1, 7, 6, 8, 6, 2, 1, 1])
        True
    """
    a = np.asarray(a)

    # Split into chunks n samples long.
    L_chunks = min(a.size, n)
    chunks = np.array_split(a, a.size//L_chunks)

    # Choose up to s chunk indices at random.
    N_chunks = min(len(chunks), s)
    rng = np.random.default_rng()
    r = rng.choice(np.arange(len(chunks)), size=N_chunks, replace=False)

    # Loop over selected chunks and count ones with correlation.
    acs = []
    for chunk in [c for i, c in enumerate(chunks) if i in r]:
        c = chunk[:L_chunks] - np.nanmean(chunk)
        autocorr = np.correlate(c, c, mode='same')
        acs.append(autocorr / (c.size * np.nanvar(c)))

    # Average the autocorrelations.
    acs = np.sum(acs, axis=0) / N_chunks

    p = acs[c.size//2 - 1]  # First non-zero lag.
    q = acs[c.size//2 - 2]  # Next non-zero lag.

    return (p >= threshold) & (q >= 0)


def zscore_outliers(z, threshold=3):
    """
    Find outliers given samples and a threshold in multiples of stdev.
    This was the fastest thing I tried. Returns -1 for outliers and 1
    for inliers (to match the sklearn outlier predictor API).

    Expect points outside:
        - 1 sd: expect 31.7 points in 100
        - 2 sd: 4.55 in 100
        - 3 sd: 2.70 in 1000
        - 4 sd: 6.3 in 100,000
        - 4.89163847 sd: 1 in 1 million
        - 5 sd: 5.7 in 10 million datapoints
        - 6 sd: 2.0 in 1 billion points

    Args:
        z (array): The samples as Z-scores.
        threshold (float): The threshold in multiples of stdev.

    Returns:
        array: Boolean array identifying outliers.

    Examples:
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> zscore_outliers(data)
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> zscore_outliers(data + [100], threshold=3)
        array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """
    z = np.squeeze(z)
    idx, = np.where((z < -threshold) | (z > threshold))
    outliers = np.full(z.shape, 1)
    outliers[idx] = -1
    return outliers


def get_outliers(a, method='iso', threshold=3):
    """
    Returns outliers in the data, considering all of the features. What counts
    as an outlier is determined by the threshold, which is in multiples of
    the standard deviation. (The conversion to 'contamination' is approximate.)

    This function requires the scikit-learn package.

    Methods: 'iso' (isolation forest), 'lof' (local outlier factor), 'svm'
    (one-class support vvector machine), or pass a function that returns
    a Boolean array of outlier flags. Note that the OneClassSVM method does
    not always yield good results in my testing. You can also pass 'any', which
    will try all three outlier detection methods and return the outliers which
    are detected by any of them, or 'all', which will return the outliers which
    are common to all three methods.

    Args:
        a (array): The data.
        method (str): The method to use. Only 'zscore' is supported.
        threshold (float): The threshold in multiples of stdev.

    Returns:
        array: The indices of the outliers.

    Examples:
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> get_outliers(3 * data)
        array([], dtype=int64)
        >>> get_outliers(3 * data + [100])
        array([33])
    """
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    expect = 1 - stdev_to_proportion(threshold)
    methods = {'iso': IsolationForest(contamination=expect).fit_predict,
               'svm': OneClassSVM(nu=expect).fit_predict,
               'lof': LocalOutlierFactor(contamination=expect, novelty=False).fit_predict,
               'mah': EllipticEnvelope(contamination=expect).fit_predict,
              }
    if method == 'any':
        results = [get_idx(func(a)==-1) for func in methods.values()]
        outliers = reduce(np.union1d, results)
    elif method == 'all':
        results = [get_idx(func(a)==-1) for func in methods.values()]
        outliers = reduce(np.intersect1d, results)
    else:
        func = methods.get(method, method)
        outliers, = np.where(func(a) == -1)
    return outliers

        
def wasserstein_ovr(a, groups=None, standardize=False):
    """
    First Wasserstein distance between each group in `a` vs the rest of `a`
    ('one vs rest' or OVR).
    
    The results are in `np.unique(a)` order.
    
    Data should be standardized for results you can compare across different
    measurements. The function does not apply standardization by default.
    
    Returns K scores for K groups.

    Args:
        a (array): The data.
        groups (array): The group labels.
        standardize (bool): Whether to standardize the data. Default False.

    Returns:
        array: The Wasserstein distance scores in `np.unique(a)` order.

    Examples:
        >>> data = [1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3]
        >>> groups = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> wasserstein_ovr(data, groups=groups, standardize=True)
        array([0.97490053, 0.1392715 , 1.11417203])
    """
    if standardize:
        a = (a - np.nanmean(a)) / np.nanstd(a)
    dists = []
    for group in iter_groups(groups):
        dist = wasserstein_distance(a[group], a[~group])
        dists.append(dist)
    return np.array(dists)


def wasserstein_ovo(a, groups=None, standardize=False):
    """
    First Wasserstein distance between each group in `a` vs each other group
    ('one vs one' or OVO).
    
    The results are in the order given by `combinations(np.unique(groups),
    r=2)`, which matches the order of `scipy.spatial.distance` metrics.
    
    Data should be standardized for results you can compare across different
    measurements. The function does not apply standardization by default.
    
    Returns K(K-1)/2 scores for K groups.

    Args:
        a (array): The data.
        groups (array): The group labels.
        standardize (bool): Whether to standardize the data. Defaults to False.

    Returns:
        array: The Wasserstein distance scores. Note that the order is the
            same as you would get from `scipy.spatial.distance` metrics. You
            can pass the result to `scipy.spatial.distance.squareform` to
            get a square matrix.

    Examples:
        >>> data = [1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3]
        >>> groups = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> wasserstein_ovo(data, groups=groups, standardize=True)
        array([0.55708601, 1.39271504, 0.83562902])
        >>> squareform(wasserstein_ovo(data, groups=groups, standardize=True))
        array([[0.        , 0.55708601, 1.39271504],
               [0.55708601, 0.        , 0.83562902],
               [1.39271504, 0.83562902, 0.        ]])
    """
    if standardize:
        a = (a - np.nanmean(a)) / np.nanstd(a)
    dists = []
    for (group_1, group_2) in combinations(np.unique(groups), r=2):
        dist = wasserstein_distance(a[groups==group_1], a[groups==group_2])
        dists.append(dist)
    return np.array(dists)


def wasserstein(X, groups=None, method='ovr', standardize=False, reducer=None):
    """
    Step over all features and apply the distance function to the groups.
    
    Method can be 'ovr', 'ovo', or a function.
    
    The function `reducer` is applied to the ovo result to reduce it to one
    value per group per feature. If you want the full array of each group
    against each other, either pass the identity function (`lambda x: x`,
    which adds an axis) or use `wasserstein_ovo()` directly, one feature at
    a time. Default function: `np.mean`.

    Args:
        X (array): The data. Must be a 2D array, or a sequence of 2D arrays.
            If the latter, then the groups are implicitly assumed to be the
            datasets in the sequence and the `groups` argument is ignored.
        groups (array): The group labels.
        method (str or func): The method to use. Can be 'ovr', 'ovo', or a
            function.
        standardize (bool): Whether to standardize the data. Default False.
        reducer (func): The function to reduce the ovo result to one value
            per group. Default: `np.mean`.

    Returns:
        array: The 2D array of Wasserstein distance scores.

    Examples:
        >>> data = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3])
        >>> groups = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> wasserstein(data.reshape(-1, 1), groups=groups, standardize=True)
        array([[0.97490053],
               [0.1392715 ],
               [1.11417203]])
        >>> wasserstein(data.reshape(-1, 1), groups=groups, method='ovo', standardize=True)
        array([[0.97490053],
               [0.69635752],
               [1.11417203]])
    """
    # If the data is a sequence of arrays, then assume the groups are the
    # datasets in the sequence and the `groups` argument is ignored.
    try:
        first = X[0]
    except KeyError:
        # Probably a DataFrame.
        first = np.asarray(X)[0]

    stacked = False
    try:
        if first.ndim == 2:
            stacked = True
    except AttributeError:
        # It's probably a 1D array or list.
        pass
    
    if stacked:
        if not is_standardized(first):
            warnings.warn('First group does not appear to be standardized.', stacklevel=2)
        groups = np.hstack([len(dataset)*[i] for i, dataset in enumerate(X)])
        X = np.vstack(X)

    # Now we can certainly treat X as a 2D array.
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array-like.")
    
    if groups is None:
        raise ValueError("Must provide a 1D array of group labels if X is a 2D array.")
    n_groups = np.unique(groups).size

    if n_groups < 2:
        raise ValueError("Must have 2 or more groups.")

    methods = {
        'ovr': wasserstein_ovr,
        'ovo': wasserstein_ovo,
    }
    func = methods.get(method, method)
    
    if reducer is None:
        reducer = np.mean

    dist_arrs = []
    for feature in X.T:
        dists = func(feature, groups=groups, standardize=standardize)
        if method == 'ovo':
            dists = squareform(dists)
            dists = dists[~np.eye(n_groups, dtype=bool)].reshape(n_groups, -1)
            dists = [reducer(d) for d in dists]
        dist_arrs.append(dists)

    return np.swapaxes(dist_arrs, 0, 1)
