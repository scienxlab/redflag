# Changelog

## 0.1.9, in progress

- More docs!

## 0.1.8, 8 July 2022

- Added Wasserstein distance comparisons for univariate and multivariate distributions. This works for either a `groups` array, or for multiple dataset splits if that's more convenient.
- Improved `get_outliers()`, removing OneClassSVM method and adding EllipticEnvelope and Mahalanobis distance.
- Added Mahalanobis distance outlier detection function to serve `get_outlier()` or be used on its own. Reproduces the results `zscore_outliers()` used to give for univariate data, so removed that.
- Added `kde_peaks()` function to find peaks in a kernel density estimate. This also needed some other functions, including `fit_kde()`, `get_kde()`, `find_large_peaks()`, and the bandwidth estimators, `bw_silverman()` and `bw_scott()`.
- Added `classes` argument to the class imbalance function, in case there are classes with no data, or to override the classes in the data.
- Fixed a bug in the `feature_importances()` function.
- Fixed a bug in the `is_continuous()` function.
- Improved the `Using_redflag.ipynb` notebook.
- Added `has_nans()`, `has_monotonic()`, and `has_flat()` functions to detect interpolation issues.
- Moved some more helper functions into utils, eg `iter_groups()`, `ecdf()`, `flatten()`, `stdev_to_proportion()` and `proportion_to_stdev()`.
- Wrote a lot more tests, coverage is now at 95%.


## 0.1.3 to 0.1.7, 9–11 February 2022

- Added `utils.has_low_distance_stdev`.
- Added `utils.has_few_samples`.
- Added `utils.is_standardized()` function to test if a feature or regression target appears to be a Z-score.
- Changed name of `clips()` function to `clipped()` to be more predictable (it goes with `is_clipped()`).
- Documentation.
- CI workflow seems to be stable.
- Mostly just a lot of flailing.


## 0.1.2, 1 February 2022

- Early release.
- Added auto-versioning.


## 0.1.1, 31 January 2022

- Early release.


## 0.1.0, 30 January 2022

- Early release.
