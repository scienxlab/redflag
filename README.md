# Redflag

[![Build and test](https://github.com/scienxlab/redflag/actions/workflows/build-test.yml/badge.svg)](https://github.com/scienxlab/redflag/actions/workflows/build-test.yml)
[![Documentation](https://github.com/scienxlab/redflag/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/scienxlab/redflag/actions/workflows/publish-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/redflag.svg)](https://pypi.org/project/redflag/)
[![PyPI versions](https://img.shields.io/pypi/pyversions/redflag.svg)](https://pypi.org/project/redflag/)
[![PyPI license](https://img.shields.io/pypi/l/redflag.svg)](https://pypi.org/project/redflag/)

🚩 `redflag` aims to be an automatic safety net for machine learning datasets. The vision is to accept input of a Pandas `DataFrame` or NumPy `ndarray` (one for each of the input `X` and target `y` in a machine learning task). `redflag` will provide an analysis of each feature, and of the target, including aspects such as class imbalance, leakage, outliers, anomalous data patterns, threats to the IID assumption, and so on. The goal is to complement other projects like `pandas-profiling` and `greatexpectations`.

⚠️ **This project is very rough and does not do much yet. The API will very likely change without warning. Please consider contributing!**


## Installation

You can install this package with `pip`:

    pip install redflag

For developers, there is a `pip` option for installing `dev` dependencies. Use `pip install redflag[dev]` to install all testing and documentation packages.


## Example

For the most part, `redflag` is currently a collection of functions. Most of the useful ones take one or more columns of data (usually a 1D or 2D NumPy array) and run a single test. For example, we can do some outlier detection: the `get_outliers()` function returns the indices of data points that are considered outliers:

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

See [the documentation](https://code.scienxlab.org/redflag), and specifically the notebook [Basic_usage.ipynb](https://github.com/scienxlab/redflag/blob/main/docs/notebooks/Basic_usage.ipynb) for several other basic examples.


## Documentation

[The documentation is online.](https://code.scienxlab.org/redflag)


## Contributing

Please see [`CONTRIBUTING.md`](https://github.com/scienxlab/redflag/blob/main/CONTRIBUTING.md). There is also a section [in the documentation](https://code.scienxlab.org/redflag) about _Development_.


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    pytest

Most of the tests are doctests, but `pytest` will run them using the settings in `setup.cfg`.
