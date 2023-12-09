# Changelog

## 0.4.2, 10 December 2023

- This is a minor release making changes to the tests and documentation in reponse to the review process for [a submission](https://joss.theoj.org/papers/e1ca575ec0c5344144f87176539ef547) to [The Journal of Open Source Software](https://joss.theoj.org) (JOSS).
- See the following issues: [#89](https://github.com/scienxlab/redflag/issues/89), [#90](https://github.com/scienxlab/redflag/issues/90), [#91](https://github.com/scienxlab/redflag/issues/91), [#92](https://github.com/scienxlab/redflag/issues/92), [#93](https://github.com/scienxlab/redflag/issues/93), [#94](https://github.com/scienxlab/redflag/issues/94) and [#95](https://github.com/scienxlab/redflag/issues/95).
- Now building and testing on Windows and MacOS as well as Linux.


## 0.4.1, 2 October 2023

- This is a minor release intended to preview new `pandas`-related features for version 0.5.0.
- Added another `pandas` Series accessor, `is_imbalanced()`.
- Added two `pandas` DataFrame accessors, `feature_importances()` and `correlation_detector()`. These are experimental features.


## 0.4.0, 28 September 2023

- `redflag` can now be installed by the `conda` package and environment manager. To do so, use `conda install -c conda-forge redflag`.
- All of the `sklearn` components can now be instantiated with `warn=False` in order to trigger a `ValueException` instead of a warning. This allows you to build pipelines that will break if a detector is triggered.
- Added `redflag.target.is_ordered()` to check if a single-label categorical target is ordered in some way. The test uses a Markov chain analysis, applying chi-squared test to the transition matrix. In general, the Boolean result should only be used on targets with several classes, perhaps at least 10. Below that, it seems to give a lot of false positives.
- You can now pass `groups` to `redflag.distributions.is_multimodal()`. If present, the modality will be checked for each group, returning a Boolean array of values (one for each group). This allows you to check a feature partitioned by target class, for example.
- Added `redflag.sklearn.MultimodalityDetector` to provide a way to check for multimodal features. If `y` is passed and is categorical, it will be used to partition the data and modality will be checked for each class.
- Added `redflag.sklearn.InsufficientDataDetector` which checks that there are at least M<sup>2</sup> records (rows in `X`), where M is the number of features (i.e. columns) in `X`.
- Removed `RegressionMultimodalDetector`. Use `MultimodalDetector` instead.


## 0.3.0, 21 September 2023

- Added some accessors to give access to `redflag` functions directly from `pandas.Series` objects, via an 'accessor'. For example, for a Series `s`, one can call `minority_classes = s.redflag.minority_classes()` instead of `redflag.minority_classes(s)`. Other functions include `imbalance_degree()`, `dummy_scores()` (see below). Probably not very useful yet, but future releases will add some reporting functions that wrap multiple Redflag functions. **This is an experimental feature and subject to change.**
- Added a Series accessor `report()` to perform a range of tests and make a small text report suitable for printing. Access for a Series `s` like `s.redflag.report()`. **This is an experimental feature and subject to change.**
- Added new documentation page for the Pandas accessor.
- Added `redflag.target.dummy_classification_scores()`, `redflag.target.dummy_regression_scores()`, which train a dummy (i.e. naive) model and compute various relevant scores (MSE and R2 for regression, F1 and ROC-AUC for classification tasks). Additionally, both `most_frequent` and `stratified` strategies are tested for classification tasks; only the `mean` strategy is employed for regression tasks. The helper function `redflag.target.dummy_scores()` tries to guess what kind of task suits the data and calls the appropriate function.
- Moved `redflag.target.update_p()` to `redflag.utils`.
- Added `is_imbalanced()` to return a Boolean depending on a threshold of imbalance degree. Default threshold is 0.5 but the best value is up for debate.
- Removed `utils.has_low_distance_stdev`.


## 0.2.0, 4 September 2023

- Moved to something more closely resembling semantic versioning, which is the main reason this is version 0.2.0.
- Builds and tests on Python 3.11 have been successful, so now supporting this version.
- Added custom 'alarm' `Detector`, which can be instantiated with a function and a warning to emit when the function returns True for a 1D array. You can easily write your own detectors with this class.
- Added `make_detector_pipeline()` which can take sequences of functions and warnings (or a mapping of functions to warnings) and returns a `scikit-learn.pipeline.Pipeline` containing a `Detector` for each function.
- Added `RegressionMultimodalDetector` to allow detection of non-unimodal distributions in features, when considered across the entire dataset. (Coming soon, a similar detector for classification tasks that will partition the data by class.)
- Redefined `is_standardized` (deprecated) as `is_standard_normal`, which implements the Kolmogorov&ndash;Smirnov test. It seems more reliable than assuming the data will have a mean of almost exactly 0 and standard deviation of exactly 1, when all we really care about is that the feature is roughly normal.
- Changed the wording slightly in the existing detector warning messages.
- No longer warning if `y` is `None` in, eg, `ImportanceDetector`, since you most likely know this.
- Some changes to `ImportanceDetector`. It now uses KNN estimators instead of SVMs as the third measure of importance; the SVMs were too unstable, causing numerical issues. It also now requires that the number of important features is less than the total number of features to be triggered. So if you have 2 features and both are important, it does not trigger.
- Improved `is_continuous()` which was erroneously classifying integer arrays with many consecutive values as non-continuous.
- Note that `wasserstein` no longer checks that the data are standardized; this check will probably return in the future, however.
- Added a `Tutorial.ipynb` notebook to the docs.
- Added a **Copy** button to code blocks in the docs.


## 0.1.10, 21 November 2022

- Added `redflag.importance.least_important_features()` and `redflag.importance.most_important_features()`. These functions are complementary (in other words, if the same threshold is used in each, then between them they return all of the features). The default threshold for importance is half the expected value. E.g. if there are 5 features, then the default threshold is half of 0.2, or 0.1. Part of [Issue 2](https://github.com/scienxlab/redflag/issues/2).
- Added `redflag.sklearn.ImportanceDetector` class, which warns if 1 or 2 features have anomalously high importance, or if some features have anomalously low importance. Part of [Issue 2](https://github.com/scienxlab/redflag/issues/2).
- Added `redflag.sklearn.ImbalanceComparator` class, which learns the imbalance present in the training data, then compares what is observed in subsequent data (evaluation, test, or production data). If there's a difference, it throws a warning. Note: it does not warn if there is imbalance present in the training data; use `ImbalanceDetector` for that.
- Added `redflag.sklearn.RfPipeline` class, which is needed to include the `ImbalanceComparator` in a pipeline (because the common-or-garden `sklearn.pipeline.Pipeline` class does not pass `y` into a transformer's `transform()` method). Also added the `redflag.sklearn.make_rf_pipeline()` function to help make pipelines with this special class. These components are straight-up forks of the code in `scikit-learn` (3-clause BSD licensed).
- Added example to `docs/notebooks/Using_redflag_with_sklearn.ipynb` to show how to use these new objects.
- Improved `redflag.is_continuous()`, which was buggy; see [Issue 3](https://github.com/scienxlab/redflag/issues/3). It still fails on some cases. I'm not sure a definitive test for continuousness (or, conversely, discreteness) is possible; it's just a heuristic.


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
