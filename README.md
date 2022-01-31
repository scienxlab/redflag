# redflag

[![Python package](https://github.com/agile-geoscience/redflag/actions/workflows/build-test.yml/badge.svg)](https://github.com/agile-geoscience/redflag/actions/workflows/build-test.yml)
[![PyPI status](https://img.shields.io/pypi/status/redflag.svg)](https://pypi.org/project/redflag//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/redflag.svg)](https://pypi.org/project/redflag//)
[![PyPI license](https://img.shields.io/pypi/l/redflag.svg)](https://pypi.org/project/redflag/)

Automatic safety net for machine learning datasets.

## Installation

You can install this package with `pip`:

    pip install redflag

### Experimental

Installing `scikit-learn` allows you to access some extra options for outlier detection.

    pip install redflag[sklearn]


## Example

Coming soon.


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    python run_tests.py


## Building

This repo uses PEP 517-style packaging. [Read more about this](https://setuptools.pypa.io/en/latest/build_meta.html) and [about Python packaging in general](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Building the project requires `build`, so first:

    pip install build

Then to build `redflag` locally:

    python -m build

The builds both `.tar.gz` and `.whl` files, either of which you can install with `pip`.


## Continuous integration

This repo has two GitHub 'workflows' or 'actions':

- Push to `main`: Run all tests on all version of Python. This is the **Run tests** workflow.
- Publish a new release: Build and upload to PyPI. This is the **Publish to PyPI** workflow. Publish using the GitHub interface, for example ([read more](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)

---

&copy; 2021 Agile Scientific, openly licenced under Apache 2.0
