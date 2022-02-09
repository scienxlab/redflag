# Redflag

[![Build and test](https://github.com/agile-geoscience/redflag/actions/workflows/build-test.yml/badge.svg)](https://github.com/agile-geoscience/redflag/actions/workflows/build-test.yml)
[![Build docs](https://github.com/agile-geoscience/redflag/actions/workflows/sphinx-docs.yml/badge.svg)](https://github.com/agile-geoscience/redflag/actions/workflows/sphinx-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/redflag.svg)](https://pypi.org/project/redflag//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/redflag.svg)](https://pypi.org/project/redflag//)
[![PyPI license](https://img.shields.io/pypi/l/redflag.svg)](https://pypi.org/project/redflag/)

🚩 `redflag` aims to be an automatic safety net for machine learning datasets. The vision i sto accept input of a Pandas `DataFrame` or NumPy `ndarray` (one for each of the input `X` and target `y` in a machine learning task). `redflag` will provide an analysis of each feature, and of the target, including aspects such as class imbalance, outliers, anomalous data patterns, threats to the IID assumption, and so on. The goal is to complement other projects like `pandas-profiling` and `greatexpectations`.

```{admonition} Work in progress
This project is very rough and does not do much yet. The API will very likely change without warning. Please consider contributing!
```


## Installation

You can install this package with `pip`:

    pip install redflag

Installing `scikit-learn` allows you to access some extra options for outlier detection.

    pip install redflag[sklearn]

For developers, there are also options for installing `tests`, `docs` and `dev` dependencies.


## Example

`redflag` is currently just a collection of functions. Most of the useful ones take a single column of data (e.g. a 1D NumPy array) and run a single test. For example, we can do some outlier detection:

```python
>>> import redflag as rf
>>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
>>> rf.has_outliers(data)
array([], dtype=int64)
>>> rf.has_outliers(3 * data + [100])
array([100])
```

See the notebook [Using_redflag.ipynb](https://github.com/agile-geoscience/redflag/blob/main/notebooks/Using_redflag.ipynb) for several other examples.


## Documentation

[The documentation is online.](https://code.agilescientific.com/redflag)


## Contributing

Please see [`CONTRIBUTING.md`](https://github.com/agile-geoscience/redflag/blob/main/CONTRIBUTING.md). There is also a section [in the documentation](https://code.agilescientific.com/redflag) about _Development_.


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    python run_tests.py

Most of the tests are run with `doctest`.
