:hide-toc:

.. container::
   :name: forkongithub

   `Fork on GitHub <https://github.com/scienxlab/redflag>`_


Redflag: safer ML by design
===========================

    | ``redflag`` is a lightweight safety net for machine
    | learning. Given a ``DataFrame`` or ``ndarray``,
    | ``redflag`` will analyse the features and the target,
    | and warn you about class imbalance, leakage, outliers,
    | anomalous data patterns, threats to the IID assumption,
    | and more.


Quick start
-----------

.. toctree::
    :caption: Quick start

Install ``redflag`` with pip or with ``conda`` from the ``conda-forge`` channel:

.. code-block:: shell

    pip install redflag

Import ``redflag`` in your Python program:

.. code-block:: python

    import redflag as rf

There are three main ways to use ``redflag``:

1. ``scikit-learn`` components for your pipelines, e.g. ``rf.ImbalanceDetector().fit_transform(X, y)``.
2. ``pandas`` accessors on Series and DataFrames, e.g. ``df['target'].redflag.imbalance_degree()``.
3. As a library of standalone functions, e.g. ``rf.imbalance_degree(y)``.

Carry on exploring with the user guide below.


User guide
----------

.. toctree::
    :maxdepth: 2
    :caption: User guide

    installation
    what_is_redflag
    _notebooks/Basic_usage.ipynb
    _notebooks/Using_redflag_with_sklearn.ipynb
    _notebooks/Using_redflag_with_Pandas.ipynb
    _notebooks/Tutorial.ipynb


API reference
-------------

.. toctree::
    :maxdepth: 2
    :caption: API reference

    redflag


Other resources
---------------

.. toctree::
    :maxdepth: 1
    :caption: Other resources

    development
    contributing
    authors
    license
    changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
    :caption: Project links
    :hidden:

    PyPI releases <https://pypi.org/project/redflag/>
    Code in GitHub <https://github.com/scienxlab/redflag>
    Issue tracker <https://github.com/scienxlab/redflag/issues>
    Community guidelines <https://scienxlab.org/community>
    Scienxlab <https://scienxlab.org>
