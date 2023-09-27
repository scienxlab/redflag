# redflag

[![Build and test](https://github.com/scienxlab/redflag/actions/workflows/build-test.yml/badge.svg)](https://github.com/scienxlab/redflag/actions/workflows/build-test.yml)
[![Documentation](https://github.com/scienxlab/redflag/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/scienxlab/redflag/actions/workflows/publish-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/redflag.svg)](https://pypi.org/project/redflag/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/redflag.svg)](https://anaconda.org/conda-forge/redflag)
[![PyPI versions](https://img.shields.io/pypi/pyversions/redflag.svg)](https://pypi.org/project/redflag/)
[![PyPI license](https://img.shields.io/pypi/l/redflag.svg)](https://pypi.org/project/redflag/)

🚩 `redflag` aims to be an automatic safety net for machine learning datasets. The vision is to accept input of a Pandas `DataFrame` or NumPy `ndarray` representing the input `X` and target `y` in a machine learning task. `redflag` will provide an analysis of each feature, and of the target, including aspects such as class imbalance, leakage, outliers, anomalous data patterns, threats to the IID assumption, and so on. The goal is to complement other projects like `pandas-profiling` and `greatexpectations`.


## Installation

You can install this package with `pip`:

    pip install redflag

For developers, there is a `pip` option for installing `dev` dependencies. Use `pip install redflag[dev]` to install all testing and documentation packages.


## Example with `sklearn`

The most useful components of `redflag` are probably the `scikit-learn` "detectors". These sit in your pipeline, look at your training and validation data, and emit warnings if something looks like it might cause a problem. For example, we can get alerted to an imbalanced target vector `y` like so:

```python
import redflag as rf
from sklearn.datasets import make_classification

X, y = make_classification(weights=[0.1])

_ = rf.ImbalanceDetector().fit(X, y)
```

This raises a warning:

```python
🚩 The labels are imbalanced by more than the threshold (0.780 > 0.400). See self.minority_classes_ for the minority classes.
```

For maximum effect, put this and other detectors in your pipeline, or use the pre-build `rf.pipeline` which contains several useful alerts.

See [the documentation](https://scienxlab.org/redflag), and specifically the notebook [Using `redflag` with `sklearn`.ipynb](https://github.com/scienxlab/redflag/blob/main/docs/notebooks/Using_redflag_with_sklearn.ipynb) for other examples.


## Example of function call

`redflag` is also a collection of functions. Most of the useful ones take one or more columns of data (usually a 1D or 2D NumPy array) and run a single test. For example, we can do some outlier detection. The `get_outliers()` function returns the indices of data points that are considered outliers:

```python
>>> import redflag as rf
>>> data = 3 * [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
>>> rf.get_outliers(data)
array([], dtype=int64)
```

That is, there are no outliers. But let's add a clear outlier: a new data record with a value of 100. The function returns the index position(s) of the outlier point(s):

```python
>>> rf.get_outliers(data + [100])
array([33])
```

See [the documentation](https://scienxlab.org/redflag), and specifically the notebook [Basic_usage.ipynb](https://github.com/scienxlab/redflag/blob/main/docs/notebooks/Basic_usage.ipynb) for several other basic examples.


## Documentation

[The documentation is online.](https://scienxlab.org/redflag)


## Contributing

Please see [`CONTRIBUTING.md`](https://github.com/scienxlab/redflag/blob/main/CONTRIBUTING.md). There is also a section [in the documentation](https://scienxlab.org/redflag) about _Development_.
