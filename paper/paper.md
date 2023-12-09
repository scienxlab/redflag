---
title: 'Redflag: machine learning safety by design'
tags:
  - Python
  - machine learning
  - data science
  - statistics
  - scikit-learn
  - pandas
  - trust
  - safety
authors:
  - name: Matt Hall
    orcid: 0000-0002-4054-8295
    affiliation: 1
affiliations:
 - name: Equinor, Bergen, Norway
   index: 1
date: 1 October 2023
bibliography: paper.bib
---

# Summary

_Redflag_ is a Python library that applies "safety by design" to machine
learning. It helps researchers and practitioners in this field ensure their
models are safe and reliable by alerting them to potential pitfalls. These
pitfalls could lead to overconfidence in the model or wildly spurious
predictions. _Redflag_ offers accessible ways for users to integrate safety
checks into their workflows by providing `scikit-learn` transformers, `pandas`
accessors, and standalone functions. These components can easily be
incorporated into existing workflows, helping identify issues and enhance the
quality and safety of predictive models.

Redflag is distributed under the [Apache 2.0
license](https://www.apache.org/licenses/LICENSE-2.0). The source code is
available [on GitHub](https://github.com/scienxlab/redflag) and includes tests
and [documentation](https://scienxlab.org/redflag/). The package can be
installed from the [Python package index](https://pypi.org/project/redflag/)
with `pip install redflag` or using
[Conda](https://anaconda.org/conda-forge/redflag) with `conda install -c
conda-forge redflag`.

# Statement of need

_Safety by design_ means to 'design out' hazardous situations from complex
machines or processes before they can do harm. The concept, also known as
_prevention through design_, has been applied to civil engineering and
industrial design for decades. Recently it has also been applied to software
engineering and, more recently still, to machine learning
[@van-gelder-etal-2021]. _Redflag_ helps machine learning researchers and
practitioners design safety into their workflows.

The practice of machine learning features a great many pitfalls that threaten
the safe application of the resulting model. These pitfalls vary in the type
and seriousness of their symptoms:

1. **Minor issues** resulting in overconfidence in the model (or, equivalently,
underperformance of the model compared to expectations), such as having
insufficient data, a few spurious data points, or failing to compute feature
interactions.
2. **Moderate issues** arising from incorrect assumptions or incorrect
application of the tools. Pitfalls include not dealing appropriately with class
imbalance, not recoginizing spatial or temporal or other correlation in the
data, or overfitting to the training or test data.
3. **Major issues** resulting in egregiously spurious predictions. Causes
include feature leakage (using features unavailable in application), using
distance-dependent algorithms on unscaled data, or forgetting to scale input
features in application.
4. **Critical issues**, especially project design and implementation issues,
that result in total failure. For example, asking the wrong question, not
writing tests or documentation, not training users of the model, or violating
ethical standards.

While some of these pathologies are difficult to check with code (especially
those in class 4, above), many of them could in principle be caught
automatically by inserting checks into the workflow that trains, evaluates, and
implements the predictive model. The goal of _Redflag_ is to provide those
checks.

In the Python machine learning world, [`pandas`](https://pandas.pydata.org/)
[@mckinney-2010] is the _de facto_ tabular data manipulation package, and
[`scikit-learn`](https://scikit-learn.org/) [@pedregosa-etal-2011] is the
preeminent prototyping and implementation framework. By integrating with these
packages by providing accessors and transformers respectively, _Redflag_ aims
to be easy to learn and adopt.

_Redflag_ offers three ways for users to insert safety checks into their
machine learning workflows:

1. **`scikit-learn` transformers** which fit directly into the pipelines that
most data scientists are already using, e.g.
`redflag.ImbalanceDetector().fit_transform(X, y)`.
2. **`pandas` accessors** on Series and DataFrames, which can be called like a
method on existing Pandas objects, e.g. `df['target'].redflag.is_imbalanced()`.
3. **Standalone functions** which the user can compose their own checks and
tests with, e.g. `redflag.is_imbalanced(y)`.

The `scikit-learn` transformers are of two kinds:

- **Detectors** check every dataset they encounter. For example,
`redflag.ClippingDetector` checks for clipped data during both model fitting
and during prediction.
- **Comparators** learn some parameter in the model fitting step, then check
subsequent data against those parameters. For example,
`redflag.DistributionComparator` learns the empirical univariate distributions
of the training features, then checks that the features in subsequent datasets
are tolerably close to these baselines.

Although the `scikit-learn` components are implemented as transformers,
subclassing `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`, they
do not transform the data. They only raise warnings (or, optionally,
exceptions) when a check fails. _Redflag_ does not attempt to fix any problems
it encounters.

There are some other packages with similar goals. For example,
[`great_expectations`](https://greatexpectations.io/) provides a full-featured
framework with a great deal of capability, especially oriented around cloud
services, and a correspondingly large API. Meanwhile,
[`pandas_dq`](https://github.com/AutoViML/pandas_dq),
[`pandera`](https://github.com/unionai-oss/pandera),
[`pandas-profiling`](https://github.com/ydataai/ydata-profiling) are all
oriented around Pandas, Spark or other DataFrame-like structures. Finally,
[`evidently`](https://github.com/evidentlyai/evidently) provides a graphical
dashboard for Jupyter. In comparison, _Redflag_ is easier to set up and use
than `great_expectations` and `pandera`, and while it is compatible with
Pandas DataFrames and Jupyter it does not depend on them.

By providing to machine learning practitioners a range of alerts and alarms,
each of which can easily be inserted into existing workflows and pipelines,
_Redflag_ aims to allow anyone to create higher quality, more trustworthy
prediction models that are safer by design.

# Acknowledgements

Thanks to the [Software Underground](https://softwareunderground.org) community
for discussion and feedback over the years.

# References
