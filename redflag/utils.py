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
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature import is_standardized


def get_idx(cond):
    idx, = np.where(cond)
    return idx


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


def sorted_unique(a):
    """
    Unique items in appearance order.

    `np.unique` is sorted, `set()` is unordered, `pd.unique()` is fast, but we
    don't have to rely on it. This does the job, and is not too slow.

    Args:
        a (array): A sequence.

    Returns:
        array: The unique items, in order of first appearance.

    Example:
        >>> sorted_unique([3, 0, 0, 1, 3, 2, 3])
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


def stdev_to_proportion(threshold: float=1) -> float:
    """
    Convert a number of standard deviations into
    the proportion of samples in the interval. For
    example, 68.27% of samples lie within ±1 stdev
    of the mean in the normal distribution.
    
    Example:
        >>> stdev_to_proportion(1)
        0.6826894921370859
        >>> stdev_to_proportion(3)
        0.9973002039367398
    """
    return 2 * scipy.stats.norm.cdf(threshold) - 1
