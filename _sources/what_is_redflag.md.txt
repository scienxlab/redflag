# 🚩 What is `redflag`?

## Overview

_Redflag_ is a Python library that applies "safety by design" to machine
learning. It helps researchers and practitioners in this field ensure their
models are safe and reliable by alerting them to potential pitfalls. These
pitfalls could lead to overconfidence in the model or wildly spurious
predictions. _Redflag_ offers accessible ways for users to integrate safety
checks into their workflows by providing `scikit-learn` transformers, `pandas`
accessors, and standalone functions. These components can easily be
incorporated into existing workflows, helping identify issues and enhance the
quality and safety of predictive models.


## Safety by design

_Safety by design_ means to 'design out' hazardous situations from complex
machines or processes before they can do harm. The concept, also known as
_prevention through design_, has been applied to civil engineering and
industrial design for decades. Recently it has also been applied to software
engineering and, more recently still, to machine learning
[@van-gelder-etal-2021]. _Redflag_ helps machine learning researchers and
practitioners design safety into their workflows.

To read more about the motivation for this package, check out
[the draft paper](https://github.com/scienxlab/redflag/blob/paper/paper/paper.md)
submitted to [JOSS](https://joss.theoj.org).


## What's in `redflag`

_Redflag_ offers three ways for users to insert safety checks into their
machine learning workflows:

1. **`scikit-learn` transformers** which fit directly into the pipelines that
most data scientists are already using, e.g.
`redflag.ImbalanceDetector().fit_transform(X, y)`.
2. **`pandas` accessors** on Series and DataFrames, which can be called like a
method on existing Pandas objects, e.g. `df['target'].redflag.is_imbalanced()`.
3. **Standalone functions** which the user can compose their own checks and
tests with, e.g. `redflag.is_imbalanced(y)`.

There are two kinds of `scikit-learn` transformer:

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
