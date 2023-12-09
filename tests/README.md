## Tests

Note that most of the tests in `redflag` are doctests. The testing code is in the docstrings of the various functions, under the 'examples' heading.

There are some pytest files in `tests` as well.

The Jupyter Notebooks in `docs/notebooks` are currently not run as part of the tests, but there is an open issue to implemement this.

Test options are in `pyproject.toml`, so to run the tests: clone the repo, install the dev dependencies (e.g. with `"pip install .[dev]`") and do this from the root directory:

    pytest


## A note about NumPy dtypes

Owing to an idiosyncracy of 64-bit Windows machines, which count a 'long' int as 32-bit not 64, I have stopped `doctest` from comparing any `dtype=int64` or similar in test outputs. This is done by the custom `doctest.OutputChecker` in `tests/conftest.py`. It only runs on Windows
machines (e.g. in the CI matrix).
