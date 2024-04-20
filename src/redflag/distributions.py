"""
Functions related to understanding distributions.

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

from typing import Optional, NamedTuple, Callable, Union
from collections import namedtuple
from itertools import combinations
import warnings

import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as ss
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from .utils import is_standard_normal
from .utils import iter_groups


DISTS = ['norm', 'cosine', 'expon', 'exponpow', 'gamma', 'gumbel_l', 'gumbel_r',
         'powerlaw', 'triang', 'trapz', 'uniform',
         ]

def best_distribution(a: ArrayLike, bins: Optional[int]=None) -> NamedTuple:
    """
    Model data by finding best fit distribution to data.

    By default, the following distributions are tried: normal, cosine,
    exponential, exponential power, gamma, left-skewed Gumbel, right-skewed
    Gumbel, power law, triangular, trapezoidal, and uniform.

    The best fit is determined by the sum of squared errors (SSE) between the
    histogram and the probability density function (PDF) of the distribution.

    Returns the best fit distribution and its parameters in a named tuple.

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


def wasserstein_ovr(a: ArrayLike, groups: ArrayLike=None, standardize: bool=False) -> np.ndarray:
    """
    First Wasserstein distance between each group in `a` vs the rest of `a`
    ('one vs rest' or OVR). The groups are provided by `groups`, which must be
    a 1D array of group labels, the same length as `a`.

    The Wasserstein distance is a measure of the distance between two
    probability distributions. It is also known as the earth mover's distance.
    This function uses the implementation in `scipy.stats.wasserstein_distance`.

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


def wasserstein_ovo(a: ArrayLike, groups: ArrayLike=None, standardize: bool=False) -> np.ndarray:
    """
    First Wasserstein distance between each group in `a` vs each other group
    ('one vs one' or OVO). The groups are provided by `groups`, which must be
    a 1D array of group labels, the same length as `a`.

    The Wasserstein distance is a measure of the distance between two
    probability distributions. It is also known as the earth mover's distance.
    This function uses the implementation in `scipy.stats.wasserstein_distance`.

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


def wasserstein(X: ArrayLike,
                groups: ArrayLike=None,
                method: str='ovr', 
                standardize: bool=False,
                reducer: Callable=None) -> np.ndarray:
    """
    Step over all features and apply the distance function to the groups.

    Method can be 'ovr', 'ovo', or a function.

    The function `reducer` is applied to the ovo result to reduce it to one
    value per group per feature. If you want the full array of each group
    against each other, either pass the identity function (`lambda x: x`,
    which adds an axis) or use `wasserstein_ovo()` directly, one feature at
    a time. Default function: `np.mean`.

    The Wasserstein distance is a measure of the distance between two
    probability distributions. It is also known as the earth mover's distance.
    This function uses the implementation in `scipy.stats.wasserstein_distance`.

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
        >>> data = [[[1], [1.22475], [-1.22475], [0], [1], [-1], [-1]], [[1], [0], [1]], [[1], [0], [-1]]]
        >>> wasserstein(data, standardize=False)
        array([[0.39754762],
               [0.71161667],
               [0.24495   ]])
    """
    # If the data is a sequence of arrays, then assume the groups are the
    # datasets in the sequence and the `groups` argument is ignored.
    try:
        first = X[0]
    except KeyError:
        # Probably a DataFrame.
        first = np.asarray(X)[0]

    stacked = False
    first = np.asarray(first)
    try:
        if first.ndim == 2:
            stacked = True
    except AttributeError:
        # It's probably a 1D array or list.
        pass

    if stacked:
        # Not sure this test makes sense any more.
        # if not is_standard_normal(first.flat):
        #     warnings.warn('First group does not appear to be standardized.', stacklevel=2)
        groups = np.hstack([len(dataset)*[i] for i, dataset in enumerate(X)])
        X = np.vstack(X)

    # Now we can treat X as a 2D array.
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


def bw_silverman(a: ArrayLike) -> float:
    """
    Calculate the Silverman bandwidth, a popular rule of thumb for kernel
    density estimation bandwidth.

    Silverman, BW (1981), "Using kernel density estimates to investigate
    multimodality", Journal of the Royal Statistical Society. Series B Vol. 43,
    No. 1 (1981), pp. 97-99.

    Args:
        a (array): The data.

    Returns:
        float: The Silverman bandwidth.

    Examples:
        >>> data = [1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3]
        >>> abs(bw_silverman(data) - 0.581810759152688) < 1e-9
        True
    """
    n, d = np.array(a).size, 1
    return np.power(n, -1 / (d + 4))


def bw_scott(a: ArrayLike) -> float:
    """
    Calculate the Scott bandwidth, a popular rule of thumb for kernel
    density estimation bandwidth.

    Args:
        a (array): The data.
    
    Returns:
        float: The Scott bandwidth.

    Examples:
        >>> data = [1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3]
        >>> abs(bw_scott(data) - 0.6162678270732356) < 1e-9
        True
    """
    n, d = np.array(a).size, 1
    return np.power(n * (d + 2) / 4, -1 / (d + 4))


def cv_kde(a: ArrayLike, n_bandwidths: int=20, cv: int=10) -> float:
    """
    Run a cross validation grid search to identify the optimal bandwidth for
    the kernel density estimation.

    Searches between half the minimum of the Silverman and Scott bandwidths,
    and twice the maximum. Checks `n_bandwidths` bandwidths, default 20.

    Args:
        a (array): The data.
        n_bandwidths (int): The number of bandwidths to try. Default 20.
        cv (int): The number of cross validation folds. Default 10.

    Returns:
        float. The optimal bandwidth.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> data = rng.normal(size=100)
        >>> cv_kde(data, n_bandwidths=3, cv=3)
        0.5212113989811242
        >>> cv_kde(rng.normal(size=(10, 10)))
        Traceback (most recent call last):
          ...
        ValueError: Data must be 1D.
    """
    a = np.asarray(a)
    if a.ndim >= 2:
        raise ValueError("Data must be 1D.")
    if not is_standard_normal(a):
        warnings.warn('Data does not appear to be standardized, the KDE may be a poor fit.', stacklevel=2)
    a = a.reshape(-1, 1)

    silverman = bw_silverman(a)
    scott = bw_scott(a)
    start = min(silverman, scott)/2
    stop = max(silverman, scott)*2
    params = {'bandwidth': np.linspace(start, stop, n_bandwidths)}
    model = GridSearchCV(KernelDensity(), params, cv=cv) 
    model.fit(a)
    return model.best_params_['bandwidth']


def fit_kde(a: ArrayLike, bandwidth: float=1.0, kernel: str='gaussian') -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a kernel density estimation to the data.

    Args:
        a (array): The data.
        bandwidth (float): The bandwidth. Default 1.0.
        kernel (str): The kernel. Default 'gaussian'.

    Returns:
        tuple: (x, kde).

    Example:
        >>> rng = np.random.default_rng(42)
        >>> data = rng.normal(size=100)
        >>> x, kde = fit_kde(data)
        >>> x[0] + 3.2124714013056916 < 1e-9
        True
        >>> kde[0] - 0.014367259502733645 < 1e-9
        True
        >>> len(kde)
        200
        >>> fit_kde(rng.normal(size=(10, 10)))
        Traceback (most recent call last):
          ...
        ValueError: Data must be 1D.
    """
    a = np.squeeze(a)
    if a.ndim >= 2:
        raise ValueError("Data must be 1D.")
    if not is_standard_normal(a):
        warnings.warn('Data does not appear to be standardized, the KDE may be a poor fit.', stacklevel=2)
    a = a.reshape(-1, 1)
    model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    model.fit(a)
    mima = 1.5 * bandwidth * np.abs(a).max()
    x = np.linspace(-mima, mima, 200).reshape(-1, 1)
    log_density = model.score_samples(x)

    return np.squeeze(x), np.exp(log_density)


def get_kde(a: ArrayLike, method: str='scott') -> tuple[np.ndarray, np.ndarray]:
    """
    Get a kernel density estimation for the data. By default, the bandwidth is
    estimated using the Scott rule of thumb. Other options are the Silverman
    rule of thumb, or cross validation (using the `cv_kde()` function).

    This function is a wrapper for `fit_kde()`, with convenient options for
    bandwidth estimation.

    Args:
        a (array): The data.
        method (str): The rule of thumb for bandwidth estimation. Must be one
            of 'silverman', 'scott', or 'cv'. Default 'scott'.

    Returns:
        tuple: (x, kde).

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> data = rng.normal(size=100)
        >>> x, kde = get_kde(data)
        >>> x[0] + 1.354649738246933 < 1e-9
        True
        >>> kde[0] - 0.162332012191087 < 1e-9
        True
        >>> len(kde)
        200
    """
    methods = {'silverman': bw_silverman, 'scott': bw_scott, 'cv': cv_kde}
    bw = methods.get(method)(a)
    return fit_kde(a, bandwidth=bw)


def find_large_peaks(x: ArrayLike, y: ArrayLike, threshold: float=0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the peaks in the array. Returns the values of x and y at the largest
    peaks, using threshold &times; max(peak amplitudes) as the cut-off. That is,
    peaks smaller than that are not returned.

    Uses `scipy.signal.find_peaks()`, with convenient options for thresholding,
    and returns the x and y values of the peaks in a named tuple.

    Args:
        x (array): The x values.
        y (array): The y values.
        threshold (float): The threshold for peak amplitude. Default 0.1.

    Returns:
        tuple: (x_peaks, y_peaks). Arrays representing the x and y values of
            the peaks.

    Examples:
        >>> x = [1, 2, 3, 4, 5, 6,  7,  8,  9, 10, 11, 12]
        >>> y = [1, 2, 3, 2, 1, 2, 15, 40, 19,  2,  1,  1]
        >>> x_peaks, y_peaks = find_large_peaks(x, y)
        >>> x_peaks
        array([8.])
        >>> y_peaks
        array([40.])
    """
    x, y = np.asarray(x), np.asarray(y)
    pos, hts = find_peaks(y, height=y)
    hts = hts['peak_heights']
    if any(hts):
        z, h = np.array([(x[p].item(), h) for p, h in zip(pos, hts) if h > threshold * hts.max()]).T
    else:
        z, h = np.array([]), np.array([])
    Peaks = namedtuple('Peaks', ['positions', 'heights'])
    return Peaks(z, h)


def kde_peaks(a: ArrayLike, method: str='scott', threshold: float=0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the peaks in the kernel density estimation. This might help you
    identify the modes in the data.

    Wraps `get_kde()` and  `find_large_peaks()` to find the peaks in the
    kernel density estimation. By default, the bandwidth is estimated using
    the Scott rule of thumb. Other options are the Silverman rule of thumb, or
    cross validation (using the `cv_kde()` function).        

    Args:
        a (array): The data.
        method (str): The rule of thumb for bandwidth estimation. Must be one
            of 'silverman', 'scott', or 'cv'. Default 'scott'.
        threshold (float): The threshold for peak amplitude. Default 0.1.

    Returns:
        tuple: (x_peaks, y_peaks). Arrays representing the x and y values of
            the peaks.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> data = np.concatenate([rng.normal(size=100)-2, rng.normal(size=100)+2])
        >>> x_peaks, y_peaks = kde_peaks(data)
        >>> x_peaks
        array([-1.67243035,  1.88998226])
        >>> y_peaks
        array([0.22014721, 0.19729456])
    """
    return find_large_peaks(*get_kde(a, method), threshold=threshold)


def is_multimodal(a: ArrayLike,
                  groups:Optional[ArrayLike]=None,
                  method: str='scott',
                  threshold: float=0.1) -> Union[bool, np.ndarray]:
    """
    Test if the data is multimodal by looking for peaks in the kernel density
    estimation. If there is more than one peak, the data are considered
    multimodal.

    If groups are passed, the data are partitioned by group and tested
    separately. The result is an array of booleans, one per group.

    Wraps `kde_peaks()` to find the peaks in the kernel density estimation.

    Args:
        a (array): The data.
        groups (array): Group labels, if the data is to be partitioned before
            testing.
        method (str): The rule of thumb for bandwidth estimation. Must be one
            of 'silverman', 'scott', or 'cv'. Default 'scott'.
        threshold (float): The threshold for peak amplitude. Default 0.1.

    Returns:
        bool or np.ndarray: True if the data appear to be multimodal. If groups
        were passed, an array with one result per group is returned.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> a = rng.normal(size=200)
        >>> is_multimodal(a)
        False
        >>> b = np.concatenate([rng.normal(size=100)-2, rng.normal(size=100)+2])
        >>> is_multimodal(b)
        True
        >>> c = np.concatenate([a, b])
        >>> is_multimodal(c, groups=[0]*200 + [1]*200)
        array([False,  True])
    """
    a = np.asarray(a)
    result = []
    with warnings.catch_warnings(record=True) as w:
        for group in iter_groups(groups):
            x, y = kde_peaks(a[group], method=method, threshold=threshold)
            result.append(len(x) > 1)
    if w:
        warnings.warn('ℹ️ Multimodality detection may not have been possible for all groups.', stacklevel=2)
    return result[0] if len(result) == 1 else np.array(result)
