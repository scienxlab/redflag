# Changelog

## 0.1.11, in development

- Coming soon...


## 0.1.10, 21 November 2022

- Added `redflag.importance.least_important_features()` and `redflag.importance.most_important_features()`. These functions are complementary (in other words, if the same threshold is used in each, then between them they return all of the features). The default threshold for importance is half the expected value. E.g. if there are 5 features, then the default threshold is half of 0.2, or 0.1. Part of [Issue 2](https://github.com/agilescientific/redflag/issues/2).
- Added `redflag.sklearn.ImportanceDetector` class, which warns if 1 or 2 features have anomalously high importance, or if some features have anomalously low importance. Part of [Issue 2](https://github.com/agilescientific/redflag/issues/2).
- Added `redflag.sklearn.ImbalanceComparator` class, which learns the imbalance present in the training data, then compares what is observed in subsequent data (evaluation, test, or production data). If there's a difference, it throws a warning. Note: it does not warn if there is imbalance present in the training data; use `ImbalanceDetector` for that.
- Added `redflag.sklearn.RfPipeline` class, which is needed to include the `ImbalanceComparator` in a pipeline (because the common-or-garden `sklearn.pipeline.Pipeline` class does not pass `y` into a transformer's `transform()` method). Also added the `redflag.sklearn.make_rf_pipeline()` function to help make pipelines with this special class. These components are straight-up forks of the code in `scikit-learn` (3-clause BSD licensed).
- Added example to `docs/notebooks/Using_redflag_with_sklearn.ipynb` to show how to use these new objects.
- Improved `redflag.is_continuous()`, which was buggy; see [Issue 3](https://github.com/agilescientific/redflag/issues/3). It still fails on some cases. I'm not sure a definitive test for continuousness (or, conversely, discreteness) is possible; it's just a heuristic.


## 0.1.9, 25 August 2022

- Added some experimental `sklearn` transformers that implement various `redflag` tests. These do not transform the data in any way, they just inspect the data and emit warnings if tests fail. The main ones are: `redflag.sklearn.ClipDetector`, `redflag.sklearn.OutlierDetector`, `redflag.sklearn.CorrelationDetector`, `redflag.sklearn.ImbalanceDetector`, and `redflag.sklearn.DistributionComparator`.
- Added tests for the `sklearn` transformers. These are in `redflag/tests/test_redflag.py` file, whereas all other tests are doctests. You can run all the tests at once with `pytest`; coverage is currently 94%.
- Added `docs/notebooks/Using_redflag_with_sklearn.ipynb` to show how to use these new objects in an `sklearn` pipeline.
- Since there's quite a bit of `sklearn` code in the `redflag` package, it is now a hard dependency. I removed the other dependencies because they are all dependencies of `sklearn`.
- Added `redflag.has_outliers()` to make it easier to check for excessive outliers in a dataset. This function only uses Mahalanobis distance and always works in a multivariate sense.
- Reorganized the `redflag.features` module into new modules: `redflag.distributions`, `redflag.outliers`, and `redflag.independence`. All of the functions are still imported into the `redflag` namespace, so this doesn't affect existing code.
- Added examples to `docs/notebooks/Basic_usage.ipynb`.
- Removed the `class_imbalance()` function, which was confusing. Use `imbalance_ratio()` instead.


## 0.1.8, 8 July 2022

- Added Wasserstein distance comparisons for univariate and multivariate distributions. This works for either a `groups` array, or for multiple dataset splits if that's more convenient.
- Improved `get_outliers()`, removing OneClassSVM method and adding EllipticEnvelope and Mahalanobis distance.
- Added Mahalanobis distance outlier detection function to serve `get_outliers()` or be used on its own. Reproduces the results `zscore_outliers()` used to give for univariate data, so removed that.
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
