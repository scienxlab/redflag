"""Test redflag"""
import pytest
import numpy as np
from sklearn.pipeline import make_pipeline

import redflag as rf


# NB Most of redflag is tested by its doctests.


def test_clip_detector():
    """
    Cannot use doctest to catch warnings.
    """
    pipe = make_pipeline(rf.ClipDetector())
    X = np.array([[2, 1], [3, 2], [4, 3], [5, 3]])
    with pytest.warns(UserWarning, match="Feature 1 may have clipped values."):
        pipe.fit_transform(X)


def test_correlation_detector():
    """
    Cannot use doctest to catch warnings.
    """
    pipe = make_pipeline(rf.CorrelationDetector())
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(size=20), np.sin(np.linspace(0, 1, 20))]).T
    with pytest.warns(UserWarning, match="Feature 1 may have correlated values."):
        pipe.fit_transform(X)
