"""
Imbalance metrics.

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

"""
This work is derived from the following reference work:
Jonathan Ortigosa-Hernandez, Inaki Inza, and Jose A. Lozano
Measuring the Class-imbalance Extent of Multi-class Problems
Pattern Recognition Letters 98 (2017)
https://doi.org/10.1016/j.patrec.2017.08.002
"""
from collections import Counter

import numpy as np

from .target import *
from .utils import *


def empirical_distribution(a):
    """
    Compute zeta and e. Equation 5 in Ortigosa-Hernandez et al. (2017).

    Args:
        a (array): A list of class labels.

    Returns:
        tuple: (zeta, e).
    """
    c = Counter(a)
    ζ = np.array([v / sum(c.values()) for v in c.values()])
    e = np.array([1 / len(c) for _ in c.values()])
    return ζ, e


def imbalance_ratio(a):
    """
    Compute the IR. Equation 6 in Ortigosa-Hernandez et al. (2017).

    Args:
        a (array): A list of class labels.

    Returns:
        float: The imbalance ratio.
    """
    ζ, _ = empirical_distribution(a)
    return max(ζ) / min(ζ)


def major_minor(a):
    """
    Returns the number of majority and minority classes.

    Args:
        a (array): A list of class labels.

    Returns:
        tuple: (maj, min), the number of majority and minority classes.

    Example:
    >>> major_minor([1, 1, 2, 2, 3, 3, 3])
    (1, 2)
    """
    ζ, e = empirical_distribution(a)
    return sum(ζ >= e), sum(ζ < e)


def divergence(method='hellinger'):
    """
    Compute the divergence between two discrete probability distributions. 

    `method` can be a string from:
        -  hellinger: Recommended.
        - euclidean: Not recommended.
        - manhattan: Recommended.
        - kl: Not recommended.
        - tv: Recommended.

    If `method` is a function, this function just hands it back.

    Args:
        ζ (array): The actual distribution.
        e (array): The expected distribution.
        method (str): The method to use.

    Returns:
        function: A divergence function.

    Reference:
        Ortigosa-Hernandez et al. (2017)
    """
    functions = {
        'hellinger': lambda x, y: np.sqrt(np.sum((np.sqrt(x) - np.sqrt(y))**2)) / np.sqrt(2),
        'euclidean': lambda x, y: np.sqrt(np.sum((x - y)**2)),
        'manhattan': lambda x, y: np.sum(np.abs(x - y)),
        'kl': lambda x, y: np.sum(x * np.log((x + 1e-12) / y)),  # Kullback-Leibler.
        'tv': lambda x, y: np.sum(np.abs(x - y)) / 2,  # Total variation.
    }
    return functions.get(method, method)


def furthest_distribution(a):
    """
    Compute the IR. Equation 6 in Ortigosa-Hernandez et al. (2017).

    Args:
        a (array): A list of class labels.

    Returns:
        array: The furthest distribution.

    Example:
        >>> furthest_distribution([3,0,0,1,2,3,2,3,2,3,1,1,2,3,3,4,3,4,3,4,])
        array([0.8, 0. , 0. , 0.2, 0. ])
    """
    ζ, e = empirical_distribution(a)
    # Construct the vector according to Eq 9.
    i = [ei if ζi >= ei else 0 for ζi, ei in zip(ζ, e)]
    # Arbitrarily increase one of the non-zero probs to sum to 1.
    i[np.argmax(i)] += 1 - sum(i)
    return np.array(i)


def imbalance_degree(a, method='manhattan'):
    r"""
    Compute IR according to Eq 8 in Ortigosa-Hernandez et al. (2017).

    .. math::
        \mathrm{ID}(\zeta) = \frac{d_\mathrm{\Delta}(\mathbf{\zeta}, \mathbf{e})}
        {d_\mathrm{\Delta}(\mathbf{\iota}_m, \mathbf{e})} + (m - 1)

    `method` can be a string from:
      - 'manhattan': Manhattan distance or L1 norm
      - 'euclidean': Euclidean distance or L2 norm
      - 'hellinger': Hellinger distance
      - 'tv': total variation distance
      - 'kl': Kullback-Leibner divergence

    It can also be a function returning a divergence. 

    Args:
        a (array): A list of class labels.
        method (str or function): The method to use.

    Returns:
        float: The imbalance degree.

    Examples:
        >>> ID = imbalance_degree(generate_data([288, 49, 288]), 'tv')
        >>> round(ID, 2)
        0.76
        >>> ID = imbalance_degree(generate_data([629, 333, 511]), 'euclidean')
        >>> round(ID, 2)
        0.3
        >>> ID = imbalance_degree(generate_data([2, 81, 61, 4]), 'hellinger')
        >>> round(ID, 2)
        1.73
        >>> ID = imbalance_degree(generate_data([2, 81, 61, 4]), 'kl')
        >>> round(ID, 2)
        1.65
    """
    ζ, e = empirical_distribution(a)
    m = sum(ζ < e)
    i = furthest_distribution(a)
    div = divergence(method)
    return (div(ζ, e) / div(i, e)) + (m - 1)


def class_imbalance(a):
    """
    Binary classification: imbalance ratio (number of expected majority class
    samples to number of expected minority samples).
    
    Multiclass classifications: imbalance degree metric, per Ortigosa-Hernandez
    et al. (2017).

    Args:
        a (array): A list of class labels.

    Returns:
        float: The imbalance ratio (binary  tasks) or imbalance degree
        (multiclass tasks).

    Examples:
        >>> class_imbalance([0, 0, 0, 1, 1, 1, 1, 1, 1])
        2.0
        >>> class_imbalance([0, 0, 0, 1, 1, 1, 1, 1, 1, 2])
        1.4
    """
    if is_binary(a):
        return imbalance_ratio(a)
    elif is_multiclass(a):
        return imbalance_degree(a)
    else:
        return None


def minority_classes(a):
    """
    Get the minority classes.

    Args:
        a (array): A list of class labels.

    Returns:
        array: The minority classes.

    Example:
        >>> minority_classes([1, 2, 2, 2, 3, 3, 3, 3, 4, 4])
        array([1, 4])
    """
    a = np.asarray(a)
    ζ, e = empirical_distribution(a)
    classes = sorted_unique(a)
    return classes[ζ < e]
