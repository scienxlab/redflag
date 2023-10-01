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

Redflag is a Python library that applies "safety by design" to machine learning. It helps researchers and practitioners in this field ensure their models are safe and reliable by alerting them to potential pitfalls. These pitfalls could lead to overconfidence in the model or wildly spurious predictions. Redflag offers accessible ways for users to integrate safety checks into their workflows by providing `scikit-learn` transformers, `pandas` accessors, and standalone functions. These components can easily be incorporated into existing workflows, helping identify issues and enhance the quality and safety of predictive models. Redflag's aim is to empower users to design and implement higher-quality models that prioritize safety from the start.

# Statement of need

_Safety by design_ means to 'design out' hazardous situations from complex machines or processes before they can do harm. The concept, also known as _prevention through design_, has been applied to civil engineering and industrial design for decades. Recently it has also been applied to software engineering and, more recently still, to machine learning [@van-gelder-etal-2021]. Redflag aims to help machine learning researchers and practitioners design safety into their workflows.

The practice of machine learning features a great many pitfalls that threaten the safe application of the resulting model. These pitfalls vary in the type and seriousness of their symptoms:

1. Minor issues resulting in overconfidence in the model (or, equivalently, underperformance of the model compared to expectations), such as having insufficient data, a few spurious data points, or failing to compute feature interactions.
2. Moderate issues arising from incorrect assumptions or incorrect application of the tools. These problems can be moderate, such as not dealing appropriately with class imbalance, not recoginizing spatial or temporal or other correlation in the data, or overfitting to the training or test data.
3. Major issues resulting in egregiously spurious predictions. Causes include feature leakage (using features unavailable in application), using distance-dependent algorithms on unscaled data, or forgetting to scale input features in application.
4. Project design and implementation issues that result in total failure, such as asking the wrong question, not writing tests or documentation, not training users of the model, or violating ethical standards.

While some of these pathologies are difficult to check with code (especially those in class 4, above), many of them could in principle be caught automatically by inserting checks into the workflow that trains, evaluates, and implements the predictive model. The goal of Redflag is to provide those checks.

In the Python machine learning world, `pandas` [@mckinney-2010; @pandas-2.1.0] is the _de facto_ tabular data manipulation package, and `scikit-learn` [@pedregosa-etal-2011; @sklearn-1.3.0] is the preeminent prototyping and implementation framework. By integrating with these packages by providing accessors and transformers respectively, Redflag aims to be as simple to include in existing workflows as possible.

Redflag offers three ways for users to insert safety checks into their machine learning workflows: 

1. `scikit-learn` transformers which fit directly into the pipelines that most data scientists are already using, e.g. `redflag.ImbalanceDetector().fit_transform(X, y)`.
2. `pandas` accessors on Series and DataFrames, which can be called like a method on existing Pandas objects, e.g. `df['target'].redflag.is_imbalanced()`, where `df` is an instance of `pd.DataFrame`.
3. Standalone functions which the user can compose their own checks and tests with, e.g. `redflag.is_imbalanced(y)`.

There are two kinds of `scikit-learn` transformer:

- **detectors** check every dataset they encounter. For example, `redflag.sklearn.ClippingDetector` checks for clipped data during both model fitting and during prediction.
- **comparators** learn some parameter in the model fitting step, then check subsequent data against those parameters. For example, `redflag.sklearn.DistributionComparator` learns the empirical univariate distributions of the training features, then checks that the features in subsequent datasets are tolerably close to these baselines (based on the Wasserstein metric $W_1$, also known as the earth mover's distance).

Although the `scikit-learn` components are implemented as transformers, subclassing `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`, they do not transform the data. They only raise warnings (or, optionally, exceptions) when a check fails. Redflag does not attempt to fix any problems it encounters.

By providing to machine learning practitioners a range of alerts and alarms, each of which can easily be inserted into existing workflows and pipelines, Redflag aims to allow anyone to create higher quality, more trustworthy prediction models that are safer by design.

# Acknowledgements

Thanks to the [Software Underground](https://softwareunderground.org) community for conversations and feedback over the years, shaping the purpose and design of this package.

# References
