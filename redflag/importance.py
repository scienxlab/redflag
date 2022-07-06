"""
Feature importance metrics.

Author: Matt Hall, agilescientific.com
Licence: Apache 2.0

Copyright 2022 Agile Scientific

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .target import is_continuous
from .utils import split_and_standardize


def feature_importances(X, y=None, task=None, random_state=None, n=3):
    """
    Measure feature importances on a task, given X and y.

    Classification tasks are assessed with logistic regression, a random
    forest, and SVM permutation importance. Regression tasks are assessed with
    lasso regression, a random forest, and SVM permutation importance. In each
    case, the `n` normalized importances with the most variance are averaged.

    Args:
        X (array): an array representing the data.
        y (array or None): an array representing the target. If None, the task
            is assumed to be an unsupervised clustering task.
        task (str or None): either 'classification' or 'regression'. If None,
            the task will be inferred from the labels and a warning will show
            the assumption being made.
        n (int): the number of tests to average. Only the n tests with the
            highest variance across features are kept.

    Returns:
        array: The importance of the features, in the order in which they
            appear in X.

    TODO:
        - Add clustering case.

    Examples:
        >>> X = [[0, 0, 0], [0, 1, 1], [0, 2, 0], [0, 3, 1], [0, 4, 0], [0, 5, 1]]
        >>> y = [5, 15, 25, 35, 45, 55]
        >>> feature_importances(X, y, task='regression', random_state=0)
        array([ 0.        ,  0.97811006, -0.19385077])
    """
    if y is None:
        task = 'clustering'
    elif task is None:
        task = 'regression' if is_continuous(y) else 'classification'

    # Split the data and ensure it is standardized.
    if task != 'clustering':
        X, X_train, X_val, y, y_train, y_val = split_and_standardize(X, y, random_state=random_state)

    # Train three models and gather the importances.
    imps = []
    if task == 'classification':
        imps.append(np.abs(LogisticRegression().fit(X, y).coef_.sum(axis=0)))
        imps.append(RandomForestClassifier(random_state=random_state).fit(X, y).feature_importances_)
        model = SVC(random_state=random_state).fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=10, scoring='f1_weighted', random_state=random_state)
        imps.append(r.importances_mean)
    elif task == 'regression':
        imps.append(np.abs(Lasso().fit(X, y).coef_))
        imps.append(RandomForestRegressor(random_state=random_state).fit(X, y).feature_importances_)
        model = SVR().fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=10, scoring='neg_mean_squared_error', random_state=random_state)
        imps.append(r.importances_mean)
    else:
        raise NotImplementedError("Cannot handle clustering problems yet.")
    imps = np.array(imps)

    # Normalize the rows by the sum of *only positive* elements.
    normalizer = np.where(imps>0, imps, 0).sum(axis=1)
    imps /= normalizer[:, None]

    # Drop imps with smallest variance and take mean of what's left.
    result = np.nanmean(sorted(imps, key=lambda row: np.std(row))[-n:], axis=0)

    return result
