"""
Functions related to understanding distributions.

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
from itertools import combinations
import warnings

import numpy as np
import scipy.stats as ss
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from .utils import is_standardized
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


def wasserstein_multi(X, groups=None):
    """
    Compute the multivariate (first) Wasserstein distance between groups.

    Returns the distance matrix for all pairwise distances ('squareform').

    Args:
        X (array): The data. Must be a 2D array, or a sequence of 2D arrays.
            If the latter, then the groups are implicitly assumed to be the
            datasets in the sequence and the `groups` argument is ignored.
        groups (array): The group labels.

    Returns:
        array: The 2D array of pairwise Wasserstein distance scores.
    """
    raise NotImplementedError()


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

    if method == 'multi':
        return wasserstein_multi(X, groups=groups)

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


def bw_silverman(a):
    """
    Calculate the Silverman bandwidth.

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


def bw_scott(a):
    """
    Calculate the Scott bandwidth.
    
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


def cv_kde(a, n_bandwidths=20, cv=10):
    """
    Run a cross validation grid search to identify the optimal bandwidth for
    the kernel density estimation.

    Args:
        a (array): The data.
        n_bandwidths (int): The number of bandwidths to try. Default 20.
        cv (int): The number of cross validation folds. Default 10.

    Returns:
        float. The optimal bandwidth.

    Examples:
        >>> data = [1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3]
        >>> abs(cv_kde(data, n_bandwidths=3, cv=3) - 0.290905379576344) < 1e-9
        True
    """
    a = np.asarray(a).reshape(-1, 1)
    silverman = bw_silverman(a)
    scott = bw_scott(a)
    start = min(silverman, scott)/2
    stop = max(silverman, scott)*2
    params = {'bandwidth': np.linspace(start, stop, n_bandwidths)}
    model = GridSearchCV(KernelDensity(), params, cv=cv) 
    model.fit(a)
    return model.best_params_['bandwidth']


def fit_kde(a, bandwidth=1.0, kernel='gaussian'):
    """
    Fit a kernel density estimation to the data.

    Args:
        a (array): The data.
        bandwidth (float): The bandwidth. Default 1.0.
        kernel (str): The kernel. Default 'gaussian'.

    Returns:
        tuple: (x, kde).

    Examples:
        >>> data = [-3, 1, -2, -2, -2, -2, 1, 2, 2, 1, 1, 2, 0, 0, 2, 2, 3, 3]
        >>> x, kde = fit_kde(data)
        >>> x[0]
        -4.5
        >>> abs(kde[0] - 0.011092399847113) < 1e-9
        True
        >>> len(kde)
        200
    """
    a = np.asarray(a)
    model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    model.fit(a.reshape(-1, 1))
    mima = 1.5 * np.abs(a).max()
    x = np.linspace(-mima, mima, 200).reshape(-1, 1)
    log_density = model.score_samples(x)
    return np.squeeze(x), np.exp(log_density)


def get_kde(a, method='scott'):
    """
    Get the kernel density estimation for the data.

    Args:
        a (array): The data.
        method (str): The rule of thumb for bandwidth estimation.
            Default 'scott'.

    Returns:
        tuple: (x, kde).

    Examples:
        >>> data = [-3, 1, -2, -2, -2, -2, 1, 2, 2, 1, 1, 2, 0, 0, 2, 2, 3, 3]
        >>> x, kde = get_kde(data)
        >>> x[0]
        -4.5
        >>> abs(kde[0] - 0.0015627693633590066) < 1e-09
        True
        >>> len(kde)
        200
    """
    methods = {'silverman': bw_silverman, 'scott': bw_scott, 'cv': cv_kde}
    bw = methods.get(method)(a)
    return fit_kde(a, bandwidth=bw)


def find_large_peaks(x, y, threshold=0.1):
    """
    Find the peaks in the array. Returns the values of x and y at the largest
    peaks, using threshold &times; max(peak amplitudes) as the cut-off. That is,
    peaks smaller than that are not returned.

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
    z, h = np.array([(x[p].item(), h) for p, h in zip(pos, hts) if h > threshold * hts.max()]).T
    Peaks = namedtuple('Peaks', ['positions', 'heights'])
    return Peaks(z, h)

def kde_peaks(a, method='scott', threshold=0.1):
    """
    Find the peaks in the kernel density estimation.

    Args:
        a (array): The data.
        method (str): The rule of thumb for bandwidth estimation.
            Default 'scott'.
        threshold (float): The threshold for peak amplitude. Default 0.1.

    Returns:
        tuple: (x_peaks, y_peaks). Arrays representing the x and y values of
            the peaks.

    Examples:
        >>> data = [-3, 1, -2, -2, -2, -2, 1, 2, 2, 1, 1, 2, 0, 0, 2, 2, 3, 3]
        >>> x_peaks, y_peaks = kde_peaks(data)
        >>> x_peaks
        array([-2.05778894,  1.74120603])
        >>> y_peaks
        array([0.15929031, 0.24708215])
    """
    return find_large_peaks(*get_kde(a, method), threshold=threshold)
