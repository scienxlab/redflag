"""
target
Functions related to understanding the target and the type of task.

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0
"""
import warnings

import numpy as np

from .utils import *


def is_regression(y):
    """
    Decide if this is most likely a regression problem.
    """
    pass

def n_classes(y):
    """
    Count the classes.

    Examples
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

    Examples
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

    Examples
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
    Decide if a single target is multiclass.

    Examples
    >>> print(is_binary([1, 1, 1]))
    False
    >>> is_binary([0, 1, 1])
    True
    >>> is_binary([1, 2, 3])
    False
    """
    return n_classes(y) == 2
