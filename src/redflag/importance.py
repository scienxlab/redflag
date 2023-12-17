"""
Feature importance metrics.

Author: Matt Hall, scienxlab.org
Licence: Apache 2.0

Copyright 2023 Redflag contributors

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
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .target import is_continuous
from .utils import split_and_standardize
from .utils import aggregate


def feature_importances(X: ArrayLike, y: ArrayLike=None,
                        task: Optional[str]=None,
                        random_state: Optional[int]=None,
    ) -> np.ndarray:
    """
    Estimate feature importances on a supervised task, given X and y.

    Classification tasks are assessed with logistic regression, a random
    forest, and KNN permutation importance. Regression tasks are assessed with
    lasso regression, a random forest, and KNN permutation importance.

    The scores from these assessments are normalized, and the normalized
    sum is returned.

    See the Tutorial in the documentation for more information.

    Args:
        X (array): an array representing the data.
        y (array or None): an array representing the target. If None, the task
            is assumed to be an unsupervised clustering task.
        task (str or None): either 'classification' or 'regression'. If None,
            the task will be inferred from the labels and a warning will show
            the assumption being made.
        random_state (int or None): the random state to use.

    Returns:
        array: The importance of the features, in the order in which they
            appear in X.

    Examples:
        >>> X = [[0, 0, 0], [0, 1, 1], [0, 2, 0], [0, 3, 1], [0, 4, 0], [0, 5, 1], [0, 7, 0], [0, 8, 1], [0, 8, 0]]
        >>> y = [5, 15, 25, 35, 45, 55, 80, 85, 90]
        >>> feature_importances(X, y, task='regression', random_state=42)
        array([0.       , 0.9831828, 0.0168172])
        >>> y = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
        >>> x0, x1, x2 = feature_importances(X, y, task='classification', random_state=42)
        >>> x1 > x2 > x0  # See Issue #49 for why this test is like this.
        True
    """
    if y is None:
        raise NotImplementedError('Unsupervised importance is not yet implemented.')

    if task is None:
        task = 'regression' if is_continuous(y) else 'classification'

    # Split the data and ensure it is standardized.
    X, X_train, X_val, y, y_train, y_val = split_and_standardize(X, y, random_state=random_state)

    # Train three models and gather the importances.
    imps: list = []
    if task == 'classification':
        imps.append(np.abs(LogisticRegression(random_state=random_state).fit(X, y).coef_.sum(axis=0)))
        imps.append(RandomForestClassifier(random_state=random_state).fit(X, y).feature_importances_)
        model = KNeighborsClassifier().fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=8, scoring='f1_weighted', random_state=random_state)
        imps.append(r.importances_mean)
    elif task == 'regression':
        imps.append(np.abs(LinearRegression().fit(X, y).coef_))
        imps.append(RandomForestRegressor(random_state=random_state).fit(X, y).feature_importances_)
        model = KNeighborsRegressor().fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=8, scoring='neg_mean_squared_error', random_state=random_state)
        imps.append(r.importances_mean)

    # Eliminate negative values and aggregate.
    imps = np.array(imps)
    imps[imps < 0] = 0
    return aggregate(imps, normalize_input=True, normalize_output=True)


def least_important_features(importances: ArrayLike,
                             threshold: Optional[float]=None) -> np.ndarray:
    """
    Returns the least important features, in order of importance (least
        important first).

    Args:
        importances (array): the importance of the features, in the order in
            which they appear in X.
        threshold (float or None): the cutoff for the importance. If None, the
            cutoff is set to half the expectation of the importance (i.e. 0.5/M
            where M is the number of features).

    Returns:
        array: The indices of the least important features.

    Examples:
        >>> least_important_features([0.05, 0.01, 0.24, 0.4, 0.3])
        array([1, 0])
        >>> least_important_features([0.2, 0.2, 0.2, 0.2, 0.2])
        array([], dtype=int64)
    """
    if threshold is None:
        threshold = 0.5 / len(importances)

    least_important: dict = {}
    for arg, imp in zip(np.argsort(importances), np.sort(importances)):
        if sum(least_important.values()) + imp > threshold:
            break
        least_important[arg] = imp

    return np.array(list(least_important)).astype(int)


def most_important_features(importances: ArrayLike,
                             threshold: Optional[float]=None) -> np.ndarray         :
    """
    Returns the indices of the most important features, in reverse order of
        importance (most important first).
 
    Args:
        importances (array): the importance of the features, in the order in
            which they appear in X.
        threshold (float or None): the cutoff for the importance. If None,
            the cutoff is set to (M-1)/M where M is the number of features.

    Returns:
        array: The indices of the most important features.

    Examples:
        >>> most_important_features([0.05, 0.01, 0.24, 0.4, 0.3])
        array([3, 4, 2])
        >>> most_important_features([0.2, 0.2, 0.2, 0.2, 0.2])
        array([4, 3, 2, 1, 0])
    """
    if threshold is None:
        threshold = 1 - 0.5 / len(importances)

    most_important: dict = {}
    args = np.argsort(importances)[::-1]
    imps = np.sort(importances)[::-1]
    for arg, imp in zip(args, imps):
        most_important[arg] = imp
        if sum(most_important.values()) > threshold:
            break

    return np.array(list(most_important)).astype(int)
