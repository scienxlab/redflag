"""
Imbalance metrics

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0

Reference:
Jonathan Ortigosa-Hernandez, Inaki Inza, and Jose A. Lozano
Measuring the Class-imbalance Extent of Multi-class Problems
Pattern Recognition Letters 98 (2017)
https://doi.org/10.1016/j.patrec.2017.08.002
"""
from collections import Counter

import numpy as np

from .tasks import *
from .utils import *

def empirical_distribution(a):
    """
    Compute zeta and e. Equation 5 in Ortigosa-Hernandez et al. (2017).
    """
    c = Counter(a)
    ζ = np.array([v / sum(c.values()) for v in c.values()])
    e = np.array([1 / len(c) for _ in c.values()])
    return ζ, e

def imbalance_ratio(a):
    """
    Compute the IR. Equation 6 in Ortigosa-Hernandez et al. (2017).
    """
    ζ, e = empirical_distribution(a)
    return max(ζ) / min(ζ)

def major_minor(a):
    """
    Returns the number of majority and minority classes.

    Example
    >>> major_minor([1, 1, 2, 2, 3, 3, 3])
    (1, 2)
    """
    ζ, e = empirical_distribution(a)
    return sum(ζ >= e), sum(ζ < e)

def hellinger(ζ, e):
    """
    Hellinger distance between discrete probability distributions.
    Recommended in Ortigosa-Hernandez et al. (2017).
    """
    return np.sqrt(np.sum((np.sqrt(ζ) - np.sqrt(e))**2)) / np.sqrt(2)

def euclidean(ζ, e):
    """
    Euclidean (L2) distance between discrete probability distributions.
    Not recommended in Ortigosa-Hernandez et al. (2017).
    """
    return np.sqrt(np.sum((ζ - e)**2))

def manhattan(ζ, e):
    """
    Manhattan (L1) distance between discrete probability distributions.
    Recommended in Ortigosa-Hernandez et al. (2017).
    """
    return np.sum(np.abs(ζ - e))

def kullback_leibler(ζ, e):
    """
    Kulllback-Leibler divergence between discrete probability distributions.
    Note that this function is not commutative.
    Not recommended in Ortigosa-Hernandez et al. (2017).
    """
    return np.sum(ζ * np.log(ζ, e))

def total_variation(ζ, e):
    """
    Total variation distance between discrete probability distributions.
    Recommended in Ortigosa-Hernandez et al. (2017).
    """
    return manhattan(ζ, e) / 2

def furthest_distribution(a):
    """
    Compute the IR. Equation 6 in Ortigosa-Hernandez et al. (2017).

    Example
    >>> furthest_distribution([3,0,0,1,2,3,2,3,2,3,1,1,2,3,3,4,3,4,3,4,])
    [0.8, 0, 0, 0.2, 0]
    """
    ζ, e = empirical_distribution(a)
    m = sum(ζ < e)
    # Construct the vector according to Eq 9.
    ι = [ei if ζi >= ei else 0 for ζi, ei in zip(ζ, e)]
    # Arbitrarily increase one of the non-zero probs to sum to 1.
    ι[np.argmax(ι)] += 1 - sum(ι)
    return ι

def imbalance_degree(a, divergence='manhattan'):
    """
    Compute IR according to Eq 8 in Ortigosa-Hernandez et al. (2017).

    `divergence` can be a string from:
      - 'manhattan': Manhattan distance or L1 norm
      - 'euclidean': Euclidean distance or L2 norm
      - 'hellinger': Hellinger distance
      - 'tv': total variation distance
      - 'kl': Kullback-Leibner divergence

    It can also be a function returning a divergence.

    Examples from Ortigosa-Hernandez et al. (2017)
    >>> ID = imbalance_degree(generate_data([288, 49, 288]), 'tv')
    >>> round(ID, 2)
    0.76
    >>> ID = imbalance_degree(generate_data([629, 333, 511]), 'euclidean')
    >>> round(ID, 2)
    0.3
    >>> ID = imbalance_degree(generate_data([2, 81, 61, 4]), 'hellinger')
    >>> round(ID, 2)
    1.73
    """
    divs = {
        'manhattan': manhattan,
        'euclidean': euclidean,
        'hellinger': hellinger,
        'tv': total_variation,
        'kl': kullback_leibler,
    }
    div = divs.get(divergence, divergence)
    ζ, e = empirical_distribution(a)
    m = sum(ζ < e)
    ι = furthest_distribution(a)
    return (div(ζ, e) / div(ι, e)) + (m - 1)

def class_imbalance(a):
    """
    Binary classification: imbalance ratio (number of expected majority
    class samples to number of expected minority samples).
    Multiclass classifications:
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

    TODO: Maybe return a dict with supports too?

    Example
    >>> minority_classes([1, 2, 2, 2, 3, 3, 3, 3, 4, 4])
    array([1, 4])
    """
    a = np.asarray(a)
    ζ, e = empirical_distribution(a)
    classes = sorted_unique(a)
    return classes[ζ < e]
