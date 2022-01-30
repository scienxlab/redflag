"""
tasks
Functions related to understanding task types.

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0
"""
def generate_data(counts):
    """
    >>> generate_data([3, 5])
    [0, 0, 0, 1, 1, 1, 1, 1]
    """
    data = [c * [i] for i, c in enumerate(counts)]
    return [item for sublist in data for item in sublist]
