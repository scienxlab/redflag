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
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .feature import is_standardized
from .target import is_regression


def feature_importance(X, y=None, task=None):
    """
    Measure feature importance on a task, given X and y.

    Classification tasks are assessed with logistic regression, a random
    forest, and SVM permutation importance. Regression tasks are assessed with
    lasso regression, a random forest, and SVM permutation importance. In each
    case, the normalized features importances are compared, the one with
    the least variance is discarded, and the other two are averaged.

    Args:
        X (array): an array representing the data.
        y (array or None): an array representing the target. If None, the task
            is assumed to be an unsupervised clustering task.
        task (str or None): either 'classification' or 'regression'. If None,
            the task will be inferred from the labels and a warning will show
            the assumption being made.

    Returns:
        array: The importance of the features, in the order in which they
            appear in X.

    TODO:
        - Tests.
    """
    if y is None:
        task = 'clustering'
    elif task is None:
        task = 'regression' if is_regression(y) else 'classification'

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

    if not is_standardized(X):
        scaler = StandardScaler().fit(X, y)
        X = scaler.transform(X)
        scaler = StandardScaler().fit(X_train, y)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    importances = []
    if task == 'classification':
        importances.append(LogisticRegression().fit(X, y).feature_importances_)
        importances.append(RandomForestClassifier().fit(X, y).feature_importances_)
        model = SVC().fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=20, random_state=0)
        importances.append(r.importances_mean)
    elif task == 'regression':
        importances.append(Lasso().fit(X, y).coef_)
        importances.append(RandomForestRegressor().fit(X, y).feature_importances_)
        model = SVR().fit(X_train, y_train)
        r = permutation_importance(model, X_val, y_val, n_repeats=20, random_state=0)
        importances.append(r.importances_mean)

    imps = np.array(importances)
    imps /= np.sum(imps, axis=1)
    return np.mean(sorted(imps, key=lambda row: np.stdev(row))[:2], axis=1)
