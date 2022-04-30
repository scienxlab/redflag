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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature import is_standardized


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
        scaler = StandardScaler().fit(X, y)
        X = scaler.transform(X)
        scaler = StandardScaler().fit(X_train, y)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val
