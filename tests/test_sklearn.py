"""Test redflag"""
import pytest
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

import redflag as rf

"""
NB Most of redflag is tested by its doctests, but doctest cannot test
   for warnings, AFAIK. Most of the tests in this file are of the sklearn API.
"""

def test_clip_detector():
    """
    Checks for clipped data. Detects clipping by looking for multiple values
    of max and/or min.
    """
    pipe = make_pipeline(rf.ClipDetector())
    X = np.array([[2, 1], [3, 2], [4, 3], [5, 3]])
    with pytest.warns(UserWarning, match="Feature 1 has samples that may be clipped."):
        pipe.fit_transform(X)

    # Does not warn:
    X = np.array([[2, 1], [3, 2], [4, 3], [5, 4]])
    pipe.fit_transform(X)


def test_correlation_detector():
    """
    Checks for data which is correlated to itself.
    """
    pipe = make_pipeline(rf.CorrelationDetector())
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(size=20), np.sin(np.linspace(0, 1, 20))]).T
    with pytest.warns(UserWarning, match="Feature 1 has samples that may be correlated."):
        pipe.fit_transform(X)


def test_simple_multimodal_detector():
    """
    Checks for features with a multimodal distribution, considered across the
    entire dataset (i.e. not per class).
    """
    pipe = make_pipeline(rf.RegressionMultimodalDetector())
    rng = np.random.default_rng(0)
    X1 = np.stack([rng.normal(size=80), rng.normal(size=80)]).T
    X2 = np.stack([rng.normal(size=80), 3 + rng.normal(size=80)]).T
    X = np.vstack([X1, X2])
    with pytest.warns(UserWarning, match="Feature 1 has samples that may be multimodally distributed."):
        pipe.fit_transform(X)


def test_custom_detector():
    """
    Checks for data which fails a user-supplied test.
    """
    has_negative = lambda x: np.any(x < 0)
    pipe = rf.make_detector_pipeline({has_negative: "are negative"})
    X = np.array([[-2, 1], [3, 2], [4, 3], [5, 4]])
    with pytest.warns(UserWarning, match="Feature 0 has samples that are negative."):
        pipe.fit_transform(X)

    pipe = rf.make_detector_pipeline([has_negative])
    with pytest.warns(UserWarning, match="Feature 0 has samples that fail"):
        pipe.fit_transform(X)

    detector = rf.Detector(has_negative)
    X = np.random.random(size=(100, 2))
    y = np.random.random(size=100) - 0.1
    assert has_negative(y)
    assert rf.is_continuous(y)
    with pytest.warns(UserWarning, match="Target 0 has samples that fail"):
        pipe.fit_transform(X, y)


def test_distribution_comparator():
    """
    Checks that the distribution of test data (i.e. transformed only) is the
    same as the distribution of the training data (i.e. fit and transformed).
    """
    pipe = make_pipeline(rf.DistributionComparator(threshold=0.5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1_000, 2))
    pipe.fit_transform(X)  # fit() never throws a warning, just learns the distribution.

    # Throws a warning on test data (tested against training statistics):
    X_test = 1 + rng.normal(size=(500, 2))
    with pytest.warns(UserWarning, match="Features 0, 1 have distributions that are different from training."):
        pipe.transform(X_test)

    # Does not warn if distribution is the same:
    X_test = rng.normal(size=(500, 2))
    pipe.fit_transform(X)


def test_univariate_outlier_detector():
    # Use a factor of 0.5 to almost guarantee that this will throw a warning.
    pipe = make_pipeline(rf.UnivariateOutlierDetector(factor=0.5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=1_000).reshape(-1, 1)
    with pytest.warns(UserWarning, match="Feature 0 has samples that are excess univariate outliers"):
        pipe.fit_transform(X)

    # Does not warn with factor of 2.5:
    pipe = make_pipeline(rf.UnivariateOutlierDetector(factor=2.5))
    pipe.fit_transform(X)


def test_multivariate_outlier_detector():
    # Use a factor of 0.5 to almost guarantee that this will throw a warning.
    pipe = make_pipeline(rf.MultivariateOutlierDetector(factor=0.5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1_000, 2))
    with pytest.warns(UserWarning, match="Dataset has more multivariate outlier samples than expected."):
        pipe.fit_transform(X)

    # Does not warn with factor of 2.5:
    pipe = make_pipeline(rf.MultivariateOutlierDetector(factor=2.5))
    pipe.fit_transform(X)


def test_outlier_detector():
    # Use a factor of 0.5 to almost guarantee that this will throw a warning.
    pipe = make_pipeline(rf.OutlierDetector(factor=0.5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1_000, 2))
    with pytest.warns(UserWarning, match="There are more outliers than expected in the training data"):
        pipe.fit_transform(X)

    # Throws a warning on test data (tested against training statistics):
    X_test = rng.normal(size=(500, 2))
    with pytest.warns(UserWarning, match="There are more outliers than expected in the data"):
        pipe.transform(X_test)

    # Does not warn with factor of 2:
    pipe = make_pipeline(rf.OutlierDetector(factor=2.0))
    pipe.fit_transform(X)


def test_imbalance_detector():
    pipe = make_pipeline(rf.ImbalanceDetector())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = rf.generate_data([20, 80])
    with pytest.warns(UserWarning, match="The labels are imbalanced"):
        pipe.fit_transform(X, y)

    # Check other method.
    pipe = make_pipeline(rf.ImbalanceDetector(method='ir', threshold=2))
    with pytest.warns(UserWarning, match="The labels are imbalanced"):
        pipe.fit_transform(X, y)

    # Does not warn with higher threshold (summary statistic for this y is 0.6):
    pipe = make_pipeline(rf.ImbalanceDetector(threshold=0.7))
    pipe.fit_transform(X, y)

    # Warns about wrong kind of y (continuous):
    y = rng.normal(size=100)
    with pytest.warns(UserWarning, match="Target y seems continuous"):
        pipe.fit_transform(X, y)

    # No warning if y is None, just skips.
    pipe.fit_transform(X)

    # Raises error because method doesn't exist:
    with pytest.raises(ValueError) as e:
        pipe = make_pipeline(rf.ImbalanceDetector(method='foo'))

    # Raises error because threshold is wrong.
    with pytest.raises(ValueError) as e:
        pipe = make_pipeline(rf.ImbalanceDetector(method='ir', threshold=0.5))

    # Raises error because threshold is wrong.
    with pytest.raises(ValueError) as e:
        pipe = make_pipeline(rf.ImbalanceDetector(method='id', threshold=2))


def test_imbalance_comparator():
    """
    The 'comparator' learns the imbalance statistics of the training set,
    then compares subsequent sets to the learned stats.
    """
    # We need to use the special redflag pipeline object, which passes
    # both X and y to `transform()`.
    pipe = rf.make_rf_pipeline(rf.ImbalanceComparator())

    # The rest is standard.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = rf.generate_data([20, 20, 20, 140])

    # Does not raise a warning because we're only fitting.
    pipe.fit(X, y)

    # Warns about different number of minority classes.
    y = rf.generate_data([20, 20, 80, 80])
    with pytest.warns(UserWarning, match="There is a different number"):
        pipe.transform(X, y)

    # Warns about wrong kind of y (continuous):
    y = rng.normal(size=100)
    with pytest.warns(UserWarning, match="Target y seems continuous"):
        pipe.fit_transform(X, y)

    # No warning if y is None, just skips:
    pipe.fit_transform(X)

    # Raises error because threshold is wrong.
    with pytest.raises(ValueError) as e:
        pipe = make_pipeline(rf.ImbalanceComparator(method='ir', threshold=0.5))

    # Raises error because threshold is wrong.
    with pytest.raises(ValueError) as e:
        pipe = make_pipeline(rf.ImbalanceComparator(method='id', threshold=2))


def test_importance_detector():
    # Raises error because method doesn't exist:
    with pytest.raises(ValueError) as e:
        pipe = make_pipeline(rf.ImportanceDetector(threshold=2))

    pipe = make_pipeline(rf.ImportanceDetector(random_state=0))

    # Warns about low importance.
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=42)
    with pytest.warns(UserWarning, match="Feature 3 has low importance"):
        pipe.fit_transform(X, y)

    # Warns about high importance.
    X, y = make_classification(n_samples=200, n_features=3, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    with pytest.warns(UserWarning, match="Feature 1 has very high importance"):
        pipe.fit_transform(X, y)

    # Warns about wrong kind of y.
    y = None
    with pytest.warns(UserWarning, match="Target y is None"):
        pipe.fit_transform(X, y)
