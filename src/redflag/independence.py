"""
Functions related to understanding row independence.

Author: Matt Hall, scienxlab.org
Licence: Apache 2.0

Copyright 2024 Redflag contributors

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
from numpy.typing import ArrayLike


def is_correlated(a: ArrayLike, n: int=20, s: int=20, threshold: float=0.1) -> bool:
    """
    Check if a dataset is auto-correlated. This function returns True if
    the 1D input array `a` appears to be correlated to itself, perhaps
    because it consists of measurements sampled at neighbouring points
    in time or space, at a spacing short enough that samples are correlated.

    If samples are correlated in this way, then the records in your dataset
    may break the IID assumption implicit in much of statistics (though not
    in specialist geostatistics or timeseries algorithms). This is not
    necessarily a big problem, but it does mean you need to be careful
    about how you split your data, for example a random split between train
    and test will leak information from train to test, because neighbouring
    samples are correlated.

    This function inspects s random chunks of n samples, averaging the
    autocorrelation coefficients across chunks. If the mean first non-zero
    lag is greater than the threshold, the array may be autocorrelated.

    See the Tutorial in the documentation for more about how to use this
    function.

    Args:
        a (array): The data.
        n (int): The number of samples per chunk.
        s (int): The number of chunks.
        threshold (float): The auto-correlation threshold.

    Returns:
        bool: True if the data are autocorrelated.

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
    acs: list = []
    for chunk in [c for i, c in enumerate(chunks) if i in r]:
        c = chunk[:L_chunks] - np.nanmean(chunk)
        autocorr = np.correlate(c, c, mode='same')
        acs.append(autocorr / (c.size * np.nanvar(c)))

    # Average the autocorrelations.
    acs = np.sum(acs, axis=0) / N_chunks

    p = acs[c.size//2 - 1]  # First non-zero lag.
    q = acs[c.size//2 - 2]  # Next non-zero lag.

    return (p >= threshold) & (q >= 0)
