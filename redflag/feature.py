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
import warnings

import numpy as np
import scipy.stats as ss
from scipy.spatial.distance import pdist


def clipped(a):
    """
    Returns the indices of values at the min and max.

    Args:
        a (array): The data.

    Returns:
        tuple: The indices of the min and max values.

    Example:
        >>> clipped([-3, -3, -2, -1, 0, 2, 3])
        (array([0, 1]), None)
    """
    min_clips, = np.where(a==np.nanmin(a))
    max_clips, = np.where(a==np.nanmax(a))
    min_clips = min_clips if len(min_clips) > 1 else None
    max_clips = max_clips if len(max_clips) > 1 else None
    return min_clips, max_clips


def is_clipped(a):
    """
    Decide if the data are likely clipped: If there are multiple
    values at the max and/or min, then the data may be clipped.

    Args:
        a (array): The data.

    Returns:
        bool: True if the data are likely clipped.

    Example:
        >>> is_clipped([-3, -3, -2, -1, 0, 2, 3])
        True
    """
    min_clips, max_clips = clipped(a)
    return (min_clips is not None) or (max_clips is not None)


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


def _find_zscore_outliers(z, threshold=3):
    """
    Find outliers given samples and a threshold in multiples of stdev.

    Returns a 

    Args:
        z (array): The samples as Z-scores.
        threshold (float): The threshold in multiples of stdev.

    Returns:
        tuple (array, float): Boolean array identifying outliers, and the ratio
            of the number of outliers to the expected number of outliers at that
            threshold. A ratio less than one indicates there are fewer outliers
            than expected; more than one means there are more. The larger the
            ratio, the worse the outlier situation.

    Examples:
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> _find_zscore_outliers(data)
        (array([], dtype=int64), 0.0)
        >>> _find_zscore_outliers(data + [100], threshold=3)
        (array([11]), 30.866528945414302)
    """
    z = np.asarray(z)
    outliers, = np.where((z < -threshold) | (z > threshold))
    n_outliers = outliers.size
    expect = 1 + ss.norm.cdf(-threshold) - ss.norm.cdf(threshold)
    ratio = (n_outliers / z.size) / expect
    return outliers, ratio


def zscore_outliers(a, sd=3, limit=4.89163847):
    """
    Find outliers using Z-scores.

    Expect points outside:
        - 1 sd: expect 31.7 points in 100
        - 2 sd: 4.55 in 100
        - 3 sd: 2.70 in 1000
        - 4 sd: 6.3 in 100,000
        - 4.89163847 sd: 1 in 1 million
        - 5 sd: 5.7 in 10 million datapoints
        - 6 sd: 2.0 in 1 billion points

    Args:
        a (array): The data.
        sd (float): The number of standard deviations to use as the threshold
            for ordinary outliers.
        limit (float): The number of standard deviations to use as the threshold
            for 'extreme' outliers.
    
    Returns:
        tuple: The ratio for outliers and extreme outliers, and the indices for
            outliers and extreme outliers.

    Examples:
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> ratio, _, _, _ = zscore_outliers(data)
        >>> ratio
        0.0
        >>> _, _, idx, _ = zscore_outliers(data + [100])
        >>> idx
        array([11])
    """
    z = (a - np.nanmean(a)) / np.nanstd(a)
    out, ratio = _find_zscore_outliers(z, threshold=sd)
    xout, xratio = _find_zscore_outliers(z, threshold=limit)
    return ratio, xratio, out, xout


def isolation_outliers(a):
    raise NotImplementedError

def local_outliers(a):
    raise NotImplementedError

def has_outliers(a, method='zscore'):
    """
    Returns significant outliers in the feature, if any (instances whose
    numbers exceeds the expected number of samples more than 4.89 standard
    deviations from the mean).

    Methods: 'zscore', 'isolation' (requires `sklearn`), 'lof' (requires
    `sklearn`), or pass a function that returns a Boolean array of outlier
    flags.

    This is going to need some work to give the functions the same API and
    return pattern.

    Args:
        a (array): The data.
        method (str): The method to use. Only 'zscore' is supported.

    Returns:
        array: The indices of the extreme outliers.

    Examples
        >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
        >>> has_outliers(data)
        array([], dtype=int64)
        >>> has_outliers(3 * data + [100])
        array([33])
    """
    a = np.asarray(a)
    methods = {'zscore': zscore_outliers,
               'isolation': isolation_outliers,
               'lof': local_outliers,
              }
    func = methods.get(method, method)
    ratio, xtreme_ratio, idx, xtreme_idx = func(a)
    return xtreme_idx


def is_standardized(a, atol=1e-5):
    """
    Returns True if the feature has zero mean and standard deviation of 1.
    In other words, if the feature appears to be a Z-score.

    Performance: this implementation was faster than np.isclose() on μ and σ,
    or comparing with z-score of entire array using np.allclose().

    Args:
        a (array): The data.
        atol (float): The absolute tolerance.

    Returns:
        bool: True if the feature appears to be a Z-score.
    """
    μ, σ = np.nanmean(a), np.nanstd(a)
    return (np.abs(μ) < atol) and (np.abs(σ - 1) < atol)


def zscore(X):
    """
    Transform array to Z-scores. If 2D, stats are computed
    per column.

    Example:
    >>> zscore([1, 2, 3, 4, 5, 6, 7, 8, 9])
    array([-1.54919334, -1.161895  , -0.77459667, -0.38729833,  0.        ,
            0.38729833,  0.77459667,  1.161895  ,  1.54919334])
    """
    return (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)


def cv(X):
    """
    Coefficient of variation, as a decimal fraction of the mean.

    Args:
        X (ndarray): The input data.

    Returns:
        float: The coefficient of variation.

    Example:
    >>> cv([1, 2, 3, 4, 5, 6, 7, 8, 9])
    0.5163977794943222
    """
    if abs(μ := np.nanmean(X, axis=0)) < 1e-12:
        warnings.warn("Mean is close to zero, coefficient of variation may not be useful.")
        μ += 1e-12
    return np.nanstd(X, axis=0) / μ


def has_low_distance_stdev(X, atol=0.1):
    """
    Returns True if the instances has a small relative standard deviation of
    distances in the feature space.

    Args:
        X (ndarray): The input data.
        atol (float): The cut-off coefficient of variation, default 0.1.

    Returns:
        bool
    """
    return cv(pdist(zscore(X))) < atol


def has_few_samples(X):
    """
    Returns True if the number of samples is less than the square of the
    number of features.

    Args:
        X (ndarray): The input data.

    Returns:
        bool

    Example:
    >>> import numpy as np
    >>> X = np.ones((100, 5))
    >>> has_few_samples(X)
    False
    >>> X = np.ones((100, 15))
    >>> has_few_samples(X)
    True
    """
    N, M = X.shape
    return N < M**2
