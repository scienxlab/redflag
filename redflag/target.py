"""
Functions related to understanding the target and the type of task.

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


def is_regression(y):
    """
    Decide if this is most likely a regression problem.
    """
    numeric = 1


def n_classes(y):
    """
    Count the classes.

    Args:
        y (array): A list of class labels.

    Returns:
        int: The number of classes.

    Examples:
        >>> n_classes([1, 1, 1])
        1
        >>> n_classes([0, 1, 1])
        2
        >>> n_classes([1, 2, 3])
        3
    """
    y_ = np.asanyarray(y)
    return np.unique(y_).size


def is_multioutput(y):
    """
    Decide if a target array is multi-output.

    Raises TypeError if y has more than 2 dimensions.

    Args:
        y (array): A list of class labels.

    Returns:
        bool: True if y has more than 1 dimensions.

    Examples:
        >>> is_multioutput([1, 2, 3])
        False
        >>> is_multioutput([[1, 2], [3, 4]])
        True
        >>> is_multioutput([[1], [2]])
        False
        >>> is_multioutput([[[1], [2]],[[3], [4]]])
        Traceback (most recent call last):
        TypeError: Target array has too many dimensions.
    """
    y_ = np.asanyarray(y)
    if y_.ndim == 1:
        return False
    elif (y_.ndim == 2):
        return y_.shape[1] > 1
    else:
        message = "Target array has too many dimensions."
        raise TypeError(message)


def is_multiclass(y):
    """
    Decide if a single target is multiclass.

    Args:
        y (array): A list of class labels.

    Returns:
        bool: True if y has more than 2 classes.

    Examples:
        >>> print(is_multiclass([1, 1, 1]))
        False
        >>> is_multiclass([0, 1, 1])
        False
        >>> is_multiclass([1, 2, 3])
        True
    """
    if n_classes(y) > 2:
        return True
    else:
        return False


def is_binary(y):
    """
    Decide if a single target is binary.

    Args:
        y (array): A list of class labels.

    Returns:
        bool: True if y has exactly 2 classes.

    Examples:
        >>> print(is_binary([1, 1, 1]))
        False
        >>> is_binary([0, 1, 1])
        True
        >>> is_binary([1, 2, 3])
        False
    """
    return n_classes(y) == 2
