## Tests

Note that most of the tests in `redflag` are doctests. The testing code is in the docstrings of the various functions, under the 'examples' heading.

There are some pytest files in `tests` as well.

The Jupyter Notebooks in `docs/notebooks` are currently not run as part of the tests, but there is an open issue to implemement this.

Test options are in `pyproject.toml`, so to run the tests: clone the repo, install the dev dependencies (e.g. with `"pip install .[dev]`") and do this from the root directory:

    pytest
