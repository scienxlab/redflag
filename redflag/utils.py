"""
Utility functions.

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
from typing import Iterable 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from scipy.optimize import fsolve
from scipy.spatial.distance import pdist


def flatten(L):
    """
    Flattens a list. For example:

    Example:
        >>> list(flatten([1, 2, [3, 4], [5, [6, 7]]]))
        [1, 2, 3, 4, 5, 6, 7]
    """
    for x in L:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def get_idx(cond):
    """
    Get the True indices of a 1D boolean array.

    Args:

        cond (array): A 1D boolean array.

    Returns:
        array: The indices of the True values.

    Example:
        >>> a = np.array([10, 20, 30, 40])
        >>> get_idx(a > 30)
        array([3])
    """
    idx, = np.where(cond)
    return idx

bool_to_index = get_idx


def index_to_bool(idx, n=None):
    """
    Convert an index to a boolean array.

    Args:
        idx (array): The indices that are True.
        n (int): The number of elements in the array.

    Returns:
        array: The boolean array.

    Example:
        >>> index_to_bool([0, 2], n=5)
        array([ True, False,  True, False, False])
    """
    if n is None:
        n = len(idx)
    return np.array([i in idx for i in range(n)])



def is_numeric(a):
    """
    Decide if a sequence is numeric.

    Args:
        a (array): A sequence.

    Returns:
        bool: True if a is numeric.

    Example:
        >>> is_numeric([1, 2, 3])
        True
        >>> is_numeric(['a', 'b', 'c'])
        False
    """
    a = np.asarray(a)
    return np.issubdtype(a.dtype, np.number)


def generate_data(counts):
    """
    Generate data from a list of counts.

    Args:
        counts (array): A sequence of class counts.

    Returns:
        array: A sequence of classes matching the counts.

    Example:
        >>> generate_data([3, 5])
        [0, 0, 0, 1, 1, 1, 1, 1]
    """
    data = [c * [i] for i, c in enumerate(counts)]
    return [item for sublist in data for item in sublist]


def ordered_unique(a):
    """
    Unique items in appearance order.

    `np.unique` is sorted, `set()` is unordered, `pd.unique()` is fast, but we
    don't have to rely on it. This does the job, and is not too slow.

    Args:
        a (array): A sequence.

    Returns:
        array: The unique items, in order of first appearance.

    Example:
        >>> ordered_unique([3, 0, 0, 1, 3, 2, 3])
        array([3, 0, 1, 2])
    """
    a = np.asarray(a)
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def split_and_standardize(X, y, random_state=None):
    """
    Split a dataset, check if it's standardized, and scale if not.

    Args:
        X (array): The training examples.
        y (array): The target or labels.
        random_state (int or None): The seed for the split.

    Returns:
        tuple of ndarray: X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state)

    if not is_standardized(X):
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    return X, X_train, X_val, y, y_train, y_val


def ecdf(arr, start='1/N', downsample=None):
    """
    Empirical CDF. No binning: the output is the length of the
    input. By default, uses the convention of starting at 1/N
    and ending at 1, but you can switch conventions.
    
    Args:
        arr (array-like): The input array.
        start (str): The starting point of the weights, must be
            'zero' (starts at 0), '1/N' (ends at 1.0), or 'mid'
            (halfway between these options; does not start at
            0 or end at 1). The formal definition of the ECDF
            uses '1/N' but the others are unbiased estimators
            and are sometimes more convenient.
        downsample (int): If you have a lot of data and want
            to sample it for performance, pass an integer. Passing 2
            will take every other sample; 3 will take every third, etc.
            
    Returns:
        tuple (ndarray, ndarray): The values and weights, aka x and y.

    Example:
        >>> ecdf([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
        >>> ecdf([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], start='mid')
        (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]))
        >>> ecdf([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], start='zero')
        (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    """
    if not downsample:  # 0 or None same as 1.
        downsample = 1
    x = np.sort(arr)[::downsample]
    if start=='1/N':
        y = np.linspace(0, 1, len(x)+1)[1:]
    elif start == 'zero':
        y = np.linspace(0, 1, len(x), endpoint=False)
    elif start == 'mid':
        y = (np.arange(len(x)) + 0.5) / len(x)
    else:
        raise ValueError("start must be '1/N', 'zero', or 'mid'.")
    return x, y


def stdev_to_proportion(threshold: float, d: float=1, n: float=1e9) -> float:
    """
    Estimate the confidence level of the scaled standard deviational
    hyperellipsoid (SDHE). This is the proportion of points whose Mahalanobis
    distance is within `threshold` standard deviations, for the given number of
    dimensions `d`.

    For example, 68.27% of samples lie within ±1 stdev of the mean in the
    univariate normal distribution. For two dimensions, `d` = 2 and 39.35% of
    the samples are within ±1 stdev of the mean.

    This is an approximation good to about 6 significant figures (depending on
    N). It uses the beta distribution to model the true distribution; for more
    about this see the following paper:
    http://poseidon.csd.auth.gr/papers/PUBLISHED/JOURNAL/pdf/Ververidis08a.pdf

    For a table of test cases see Table 1 in:
    https://doi.org/10.1371/journal.pone.0118537

    Args:
        threshold (float): The number of standard deviations (or 'magnification
            ratio').
        d (float): The number of dimensions.
        n (float): The number of instances; just needs to be large for a
            proportion with decent precision.

    Returns:
        float. The confidence level.

    Example:
        >>> stdev_to_proportion(1)  # Exact result: 0.6826894921370859
        0.6826894916531445
        >>> stdev_to_proportion(3)  # Exact result: 0.9973002039367398
        0.9973002039633309
        >>> stdev_to_proportion(1, d=2)
        0.39346933952920327
        >>> stdev_to_proportion(5, d=10)
        0.9946544947734935
    """
    return beta.cdf(x=1/n, a=d/2, b=(n-d-1)/2, scale=1/threshold**2)


def proportion_to_stdev(p: float, d: float=1, n: float=1e9) -> float:
    """
    The inverse of `stdev_to_proportion`.

    Estimate the 'magnification ratio' (number of standard deviations) of the
    scaled standard deviational hyperellipsoid (SDHE) at the given confidence
    level and for the given number of dimensions, `d`.

    This tells us the number of standard deviations containing the given
    proportion of instances. For example, 80% of samples lie within ±1.2816
    standard deviations.

    For more about this and a table of test cases (Table 2) see:
    https://doi.org/10.1371/journal.pone.0118537

    Args:
        p (float): The confidence level as a decimal fraction, e.g. 0.8.
        d (float): The number of dimensions. Default 1 (the univariate Gaussian
            distribution).
        n (float): The number of instances; just needs to be large for a
            proportion with decent precision. `Default 1e9`.

    Returns:
        float. The estimated number of standard deviations ('magnification ratio').

    Examples:
        >>> proportion_to_stdev(0.99, d=1)
        2.575829302496098
        >>> proportion_to_stdev(0.90, d=5)
        3.039137525465009
        >>> stdev_to_proportion(proportion_to_stdev(0.80, d=1))
        0.8000000000000003
    """
    func = lambda r_, d_, n_: stdev_to_proportion(r_, d_, n_) - p
    r_hat , = fsolve(func, x0=2, args=(d, n))
    return r_hat


def is_standardized(a, atol=1e-5):
    """
    Returns True if the feature has zero mean and standard deviation of 1.
    In other words, if the feature appears to be a Z-score.

    Note that if a dataset was standardized using the mean and stdev of
    another dataset (for example, a training set), then the test set will
    not itself have a mean of zero and stdev of 1.

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
    min_clips = get_idx(a==np.nanmin(a))
    max_clips = get_idx(a==np.nanmax(a))
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


def iter_groups(groups):
    """
    Allow iterating over groups, getting boolean array for each.
    
    Equivalent to `(groups==group for group in np.unique(groups))`.

    Args:
        groups (array): The group labels.

    Yields:
        array: The boolean mask array for each group.
    
    Example:
    >>> for group in iter_groups([1, 1, 1, 2, 2]):
    ...     print(group)
    [ True  True  True False False]
    [False False False  True  True]
    """
    for group in np.unique(groups):
        yield groups == group


def has_nans(a):
    """
    Returns the indices of any NaNs.

    Args:
        a (array): The data, a 1D array.

    Returns:
        ndarray: The indices of any NaNs.

    Example:
        >>> has_nans([1, 2, 3, 4, 5, 6, 7, 8, 9])
        array([], dtype=int64)
        >>> has_nans([1, 2, np.nan, 4, 5, 6, 7, 8, 9])
        array([2])
    """
    return np.nonzero(np.isnan(a))[0]


def consecutive(a, stepsize=1):
    """
    Splits an array into groups of consecutive values.
    
    Args:
        data (array): The data.
        stepsize (int): The step size.

    Returns:
        list of arrays.
    
    Example:
    >>> consecutive([0, 0, 1, 2, 3, 3])
    [array([0]), array([0, 1, 2, 3]), array([3])]
    """
    return np.split(a, get_idx(np.diff(a) != stepsize) + 1)


def has_flat(a, tolerance=3):
    """
    Returns True if the array has any flat values.

    Args:
        a (array): The data, a 1D array.
        tolerance (int): The maximum length of a 'flat' that will be allowed.

    Returns:
        ndarray: The indices of any flat intervals.

    Example:
        >>> has_flat([1, 2, 3, 4, 5, 6, 7, 8, 9])
        array([], dtype=int64)
        >>> has_flat([1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9], tolerance=3)
        array([], dtype=int64)
        >>> has_flat([1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 8, 9], tolerance=3)
        array([4, 5, 6, 7])
    """
    zeros = get_idx(np.diff(a) == 0)
    flats = [list(x)+[x[-1]+1] for x in consecutive(zeros) if x.size >= tolerance]
    return np.array(list(flatten(flats)), dtype=int)


def has_monotonic(a, tolerance=3):
    """
    Returns True if the array is monotonic.

    Args:
        a (array): The data, a 1D array.
        tolerance (int): The maximum length of a monotonic interval that will
            be allowed.

    Returns:
        ndarray: The indices of any monotonic intervals.

    Example:
        >>> has_monotonic([1, 1, 1, 1, 2, 2, 2, 2])
        array([], dtype=int64)
        >>> has_monotonic([1, 1, 1, 2, 3, 4, 4, 4])
        array([], dtype=int64)
        >>> has_monotonic([1, 1, 1, 2, 3, 4, 5, 5, 5])
        array([2, 3, 4, 5, 6])
        >>> has_monotonic([1, 1, 1, 2, 3, 4, 5, 5, 5])
        array([2, 3, 4, 5, 6])
    """
    a = np.diff(a)
    zeros = get_idx(np.diff(a) == 0)
    flats = [list(x)+[x[-1]+1, x[-1]+2] for x in consecutive(zeros) if x.size >= tolerance]
    return np.array(list(flatten(flats)), dtype=int)
