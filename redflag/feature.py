"""
feature
Functions related to understanding features.

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0
"""
import numpy as np
import scipy.stats as ss

def clips(a):
    """
    Returns the indices of values at the min and max.

    Example
    >>> clips([-3, -3, -2, -1, 0, 2, 3])
    (array([0, 1]), None)
    """
    min_clips, = np.where(a==np.nanmin(a))
    max_clips, = np.where(a==np.nanmax(a))
    min_clips = min_clips if len(min_clips) > 1 else None
    max_clips = max_clips if len(max_clips) > 1 else None
    return min_clips, max_clips

def is_clipped(a):
    """
    Decide if the data are likely clipped: If there are multiple
    values at the max and/or min, then the data may be clipped.

    Example
    >>> is_clipped([-3, -3, -2, -1, 0, 2, 3])
    True
    """
    min_clips, max_clips = clips(a)
    return (min_clips is not None) or (max_clips is not None)

DISTS = [
    'norm',
    'cosine',
    'expon',
    'exponpow',
    'gamma',
    'gumbel_l',
    'gumbel_r',
    'powerlaw',
    'triang',
    'trapz',
    'uniform',
]

def best_distribution(a, bins=None):
    """
    Model data by finding best fit distribution to data.
    """
    if bins is None:
        bins = min(max(20, len(a) // 100), 200)
    n, x = np.histogram(a, bins=bins, density=True)
    x = (x[1:] + x[:-1]) / 2

    dists = [getattr(ss, d) for d in DISTS]

    best_dist = None
    best_params = None
    best_sse = np.inf

    for dist in dists:
        *args, μ, σ = dist.fit(a)
        n_pred = dist.pdf(x, loc=μ, scale=σ, *args)
        sse = np.sum((n - n_pred)**2)
        if 0 < sse < best_sse:
            best_dist = dist
            best_params = args + [μ] + [σ]
            best_sse = sse

    return best_dist.name, best_params, best_sse

def is_correlated(a, n=20, s=20, threshold=0.1):
    """
    Check if a dataset is correlated. Uses s chunks of n samples.
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
    trials = []
    acs = []
    for chunk in [c for i, c in enumerate(chunks) if i in r]:
        c = chunk[:L_chunks] - np.nanmean(chunk)
        autocorr = np.correlate(c, c, mode='same')
        acs.append(autocorr / (c.size * np.nanvar(c)))

    # Average the autocorrelations.
    acs = np.sum(acs, axis=0) / N_chunks

    p = acs[c.size//2 - 1]  # First non-zero lag.
    q = acs[c.size//2 - 2]  # Next non-zero lag.

    return (p >= threshold) & (q >= 0)

def _find_zscore_outliers(z, threshold):
    """
    Find outliers given samples and a threshold in multiples of stdev.

    Returns a Boolean array identifying outliers, and the ratio of
    the number of outliers to the expected number of outliers at that
    threshold. A ratio less than one indicates there are fewer outliers
    than expected; more than one means there are more. The larger the
    ratio, the worse the outlier situation.
    """
    outliers, = np.where((z < -threshold) | (z > threshold))
    n_outliers = outliers.size
    expect = 1 + ss.norm.cdf(-threshold) - ss.norm.cdf(threshold)
    ratio = (n_outliers / z.size) / expect
    return outliers, ratio

def zscore_outliers(a, sd=3, limit=4.89163847):
    """
    Find outliers using Z-scores.

    Example
    >>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
    >>> ratio, *_ = zscore_outliers(data)
    >>> ratio
    0.0
    >>> *_, idx, _ = zscore_outliers(data + [100])
    >>> idx
    array([11])
    """
    # Expect points outside:
    # 1 sd: expect 31.7 points in 100
    # 2 sd: 4.55 in 100
    # 3 sd: 2.70 in 1000
    # 4 sd: 6.3 in 100,000
    # 4.89163847 sd: 1 in 1 million
    # 5 sd: 5.7 in 10 million datapoints
    # 6 sd: 2.0 in 1 billion points
    z = (a - np.nanmean(a)) / np.nanstd(a)
    out, ratio = _find_zscore_outliers(z, threshold=sd)
    xout, xratio = _find_zscore_outliers(z, threshold=limit)
    return ratio, xratio, out, xout

def isolation_outliers(a):
    raise NotImplementedError

def local_outliers(a):
    raise NotImplementedError

def has_outliers(a, method='zscore'):
    """
    Returns significant outliers in the feature, if any (instances
    whose numbers exceeds the expected number of samples more than
    4.89 standard deviations from the mean).

    Methods: zscore, isolation (requires sklearn) or pass a function
    that returns a Boolean array of outlier flags.

    This is going to need some work to give the functions the same
    API and return pattern.
    """
    a = np.asarray(a)
    methods = {'zscore': zscore_outliers,
               'isolation': isolation_outliers,
               'lof': local_outliers,
              }
    func = methods.get(method, method)
    ratio, xtreme_ratio, idx, xtreme_idx = func(a)
    return a[xtreme_idx]
