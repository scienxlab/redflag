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


def feature_importances(X, y=None, n=3, task=None, random_state=None, standardize=True):
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
        n (int): the number of tests to average. Only the n tests with the
            highest variance across features are kept.
        task (str or None): either 'classification' or 'regression'. If None,
            the task will be inferred from the labels and a warning will show
            the assumption being made.
        random_state (int or None): the random state to use.
        standardize (bool): whether to standardize the data. Default is True.

    Returns:
        array: The importance of the features, in the order in which they
            appear in X.

    Examples:
        >>> X = [[0, 0, 0], [0, 1, 1], [0, 2, 0], [0, 3, 1], [0, 4, 0], [0, 5, 1]]
        >>> y = [5, 15, 25, 35, 45, 55]
        >>> feature_importances(X, y, task='regression', random_state=0)
        array([ 0.        ,  0.97811006, -0.19385077])
        >>> y = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> feature_importances(X, y, task='classification', random_state=0)
        array([ 0.        ,  0.89013985, -0.55680651])
    """
    if y is None:
        raise NotImplementedError('Unsupervised importance is not yet implemented.')

    if task is None:
        task = 'regression' if is_continuous(y) else 'classification'

    # Split the data and ensure it is standardized.
    if standardize:
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

    imps = np.array(imps)

    # Normalize the rows by the sum of *only positive* elements.
    normalizer = np.where(imps>0, imps, 0).sum(axis=1)
    imps /= normalizer[:, None]

    # Drop imps with smallest variance and take mean of what's left.
    return np.nanmean(sorted(imps, key=lambda row: np.std(row))[-n:], axis=0)


def least_important_features(importances, threshold=None):
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

    least_important = {}
    for arg, imp in zip(np.argsort(importances), np.sort(importances)):
        if sum(least_important.values()) + imp > threshold:
            break
        least_important[arg] = imp

    return np.array(list(least_important)).astype(int)


def most_important_features(importances, threshold=None):
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

    most_important = {}
    args = np.argsort(importances)[::-1]
    imps = np.sort(importances)[::-1]
    for arg, imp in zip(args, imps):
        most_important[arg] = imp
        if sum(most_important.values()) > threshold:
            break

    return np.array(list(most_important)).astype(int)
