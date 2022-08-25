# Redflag

[![Build and test](https://github.com/agilescientific/redflag/actions/workflows/build-test.yml/badge.svg)](https://github.com/agilescientific/redflag/actions/workflows/build-test.yml)
[![PyPI version](https://img.shields.io/pypi/v/redflag.svg)](https://pypi.org/project/redflag/)
[![PyPI versions](https://img.shields.io/pypi/pyversions/redflag.svg)](https://pypi.org/project/redflag/)
[![PyPI license](https://img.shields.io/pypi/l/redflag.svg)](https://pypi.org/project/redflag/)

🚩 `redflag` aims to be an automatic safety net for machine learning datasets. The vision is to accept input of a Pandas `DataFrame` or NumPy `ndarray` (one for each of the input `X` and target `y` in a machine learning task). `redflag` will provide an analysis of each feature, and of the target, including aspects such as class imbalance, outliers, anomalous data patterns, threats to the IID assumption, and so on. The goal is to complement other projects like `pandas-profiling` and `greatexpectations`.

```{admonition} Work in progress
This project is very rough and does not do much yet. The API will very likely change without warning. Please consider contributing!
```


## Installation

You can install this package with `pip`:

    pip install redflag

For developers, there are `pip` options for installing `tests`, `docs` and `dev` dependencies, e.g. `pip install redflag[dev]` to install all testing and documentation packages.


## Example

`redflag` is currently just a collection of functions. Most of the useful ones take a single column of data (e.g. a 1D NumPy array) and run a single test. For example, we can do some outlier detection. The `get_outliers()` function returns the indices of data points that are considered outliers:

```python
>>> import redflag as rf
>>> data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]
>>> rf.get_outliers(data)
array([], dtype=int64)
>>> rf.get_outliers(3 * data + [100])
array([33])
```

See [the documentation](https://code.agilescientific.com/redflag), and specifically the notebook [Basic_usage.ipynb](https://github.com/agilescientific/redflag/blob/main/docs/notebooks/Basic_usage.ipynb) for several other basic examples.


## Documentation

[The documentation is online.](https://code.agilescientific.com/redflag)


## Contributing

Please see [`CONTRIBUTING.md`](https://github.com/agilescientific/redflag/blob/main/CONTRIBUTING.md). There is also a section [in the documentation](https://code.agilescientific.com/redflag) about _Development_.


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    pytest

Most of the tests are doctests, but `pytest` will run them using the settings in `setup.cfg`.
