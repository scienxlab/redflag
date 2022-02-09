# Redflag: An Entrance Exam for Data

🚩 `redflag` aims to be an automatic safety net for machine learning datasets. The vision i sto accept input of a Pandas `DataFrame` or NumPy `ndarray` (one for each of the input `X` and target `y` in a machine learning task). `redflag` will provide an analysis of each feature, and of the target, including aspects such as class imbalance, outliers, anomalous data patterns, threats to the IID assumption, and so on. The goal is to complement other projects like `pandas-profiling` and `greatexpectations`.

```{admonition} Work in progress
This project is very rough and does not do much yet. The API will very likely change without warning. Please consider contributing!
```
----

```{toctree}
---
maxdepth: 2
caption: User Guide
---

installation
basic_usage
examples
development
contributing
authors
changelog

```
----

```{toctree}
---
maxdepth: 2
caption: API Reference
---

api
```
----

# Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
