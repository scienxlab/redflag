"""
utils
Functions related to understanding task types.

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0
"""
import numpy as np

def generate_data(counts):
    """
    Generate data from a list of counts.

    Example
    >>> generate_data([3, 5])
    [0, 0, 0, 1, 1, 1, 1, 1]
    """
    data = [c * [i] for i, c in enumerate(counts)]
    return [item for sublist in data for item in sublist]

def sorted_unique(a):
    """
    Unique items in appearance order.

    np.unique is sorted, set() is unordered.
    pd.unique() is fast, but we don't have to rely on it.
    This does the job, and is not too slow.

    Example
    >>> sorted_unique([3, 0, 0, 1, 3, 2, 3])
    array([3, 0, 1, 2])
    """
    a = np.asarray(a)
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]
