"""Test Pandas accessors."""
import pytest
import pandas as pd
from redflag.pandas import SeriesAccessor, DataFrameAccessor


c = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 3, 3])
r = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def test_dummy_scores():
    c_scores = c.redflag.dummy_scores()
    r_scores = r.redflag.dummy_scores()

    assert c_scores['most_frequent']['roc_auc'] == 0.5
    assert r_scores['mean']['mean_squared_error'] - 0.08249999999999999 < 1e-12


def test_imbalance_metrics():
    minorities = c.redflag.minority_classes()
    assert 2 in minorities and 3 in minorities

    imb_degree = c.redflag.imbalance_degree()
    assert imb_degree - 1.25 < 1e-9


def test_warnings():
    with pytest.warns(UserWarning, match="The Series does not seem categorical."):
        r.redflag.minority_classes()
    with pytest.warns(UserWarning, match="The Series does not seem categorical."):
        r.redflag.imbalance_degree()
