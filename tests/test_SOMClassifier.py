"""Test for susi.SOMClassifier

Usage:
python -m pytest tests/test_SOMClassifier.py

"""
import pytest
import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import susi

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


@pytest.mark.parametrize("n_rows,n_columns,do_class_weighting,random_state", [
    (2, 2, True, 3),
    (2, 2, False, 3),
    (10, 20, True, 3),
])
def test_init_super_som(n_rows, n_columns, do_class_weighting, random_state):
    som = susi.SOMClassifier(
        n_rows=n_rows,
        n_columns=n_columns,
        do_class_weighting=do_class_weighting,
        random_state=random_state)

    som.X_ = X_train
    som.y_ = y_train
    som.init_estimator()
    som.som_unsupervised()
    som.set_bmus(som.X_)
    som.init_super_som()


@pytest.mark.parametrize(
    ("n_rows,n_columns,learningrate,dist_weight_matrix,random_state,"
     "class_weight,expected"), [
        (2, 2, 0.3, np.array([[1.3, 0.4], [2.1, 0.2]]), 3, 1., None),
        (2, 2, 1e-9, np.array([[1.3, 0.4], [2.1, 0.2]]), 3, 1.,
         np.array([[False, False], [False, False]])),
     ])
def test_change_class_proba(n_rows, n_columns, learningrate,
                            dist_weight_matrix, random_state, class_weight,
                            expected):
    som = susi.SOMClassifier(n_rows=n_rows, n_columns=n_columns,
                             random_state=random_state)
    # som.classes_ = [0, 1, 2]
    # som.class_weights_ = [1., 1., 1.]
    new_som_array = som.change_class_proba(learningrate, dist_weight_matrix,
                                           class_weight)
    assert(new_som_array.shape == (n_rows, n_columns, 1))
    assert(new_som_array.dtype == bool)
    if expected is not None:
        assert np.array_equal(new_som_array, expected.reshape((2, 2, 1)))


@pytest.mark.parametrize(
    ("n_rows,n_columns,learningrate,dist_weight_matrix,som_array,"
     "random_state,true_vector,expected"), [
        (2, 2, 0.3, np.array([[1.3, 0.4], [2.1, 0.2]]).reshape(2, 2, 1),
         np.array([[1.3, 0.4], [2.1, 0.2]]).reshape(2, 2, 1), 3,
         np.array([1]), None),
     ])
def test_modify_weight_matrix_supervised(
        n_rows, n_columns, learningrate, dist_weight_matrix, som_array,
        random_state, true_vector, expected):
    som = susi.SOMClassifier(
        n_rows=n_rows,
        n_columns=n_columns,
        random_state=random_state)
    som.classes_ = [0, 1, 2]
    som.class_weights_ = [1., 1., 1.]
    new_som = som.modify_weight_matrix_supervised(
        som_array, learningrate, dist_weight_matrix, true_vector)
    assert(new_som.shape == (n_rows, n_columns, 1))
