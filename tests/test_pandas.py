"""Test Pandas accessors."""
import pytest
import pandas as pd
from redflag.pandas import null_decorator, SeriesAccessor


c = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 3, 3])
r = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 3.0])


def test_null_decorator():
    @null_decorator('foo')
    def f():
        return None
    assert f() is None


def test_dummy_scores():
    c_scores = c.redflag.dummy_scores(random_state=42)
    r_scores = r.redflag.dummy_scores(random_state=42)

    assert c_scores['roc_auc'] - 0.6801587301587301 < 1e-12
    assert r_scores['mean_squared_error'] - 0.5710743801652893 < 1e-12


def test_imbalance():
    assert c.redflag.is_imbalanced(threshold=0.24, method='tv')

    minorities = c.redflag.minority_classes()
    assert 2 in minorities and 3 in minorities

    imb_degree = c.redflag.imbalance_degree()
    assert imb_degree - 1.25 < 1e-9


def test_is_ordered():
    assert c.redflag.is_ordered()
    with pytest.raises(ValueError, match="Cannot check order of continuous data."):
        r.redflag.is_ordered()


def test_warnings():
    with pytest.warns(UserWarning, match="The Series does not seem categorical."):
        r.redflag.minority_classes()
    with pytest.warns(UserWarning, match="The Series does not seem categorical."):
        r.redflag.imbalance_degree()


def test_series_categorical_report():
    report_c = c.redflag.report()
    assert 'Categorical' in report_c


def test_series_continuous_report():
    report_r = r.redflag.report()
    assert 'Continuous' in report_r


def test_feature_importances_docstring():
    s = pd.DataFrame([c, r]).redflag.feature_importances.__doc__
    assert s.strip().startswith("Estimate feature importances on a supervised task, given X and y.")
