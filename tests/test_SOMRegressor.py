"""Test for susi.SOMRegressor

Usage:
python -m pytest tests/test_SOMRegressor.py

"""
import pytest
import os
import sys
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import susi

# define test dataset
boston = load_boston()
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.5, random_state=42)
# preprocessing
scaler =  StandardScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test = scaler.transform(X_test_orig)


@pytest.mark.parametrize("n_rows,n_columns", [
    (10, 10),
    (12, 15),
])
def test_som_regressor_init(n_rows, n_columns):
    som_reg = susi.SOMRegressor(
        n_rows=n_rows, n_columns=n_columns)
    assert som_reg.n_rows == n_rows
    assert som_reg.n_columns == n_columns

@pytest.mark.parametrize("n_rows,n_columns,train_mode_supervised,random_state", [
    (3, 3, "online", 42),
    (3, 3, "batch", 42),
])
def test_predict(n_rows, n_columns, train_mode_supervised, random_state):
    som_reg = susi.SOMRegressor(
        n_rows=n_rows, n_columns=n_columns,
        train_mode_supervised=train_mode_supervised, random_state=random_state)

    # TODO remove after implementation of supervised batch mode
    if train_mode_supervised == "batch":
        with pytest.raises(Exception):
            som_reg.fit(X_train, y_train)
    else:
        som_reg.fit(X_train, y_train)
        y_pred = som_reg.predict(X_test)
        assert(y_pred.shape == y_test.shape)

@pytest.mark.parametrize("estimator", [
    (susi.SOMRegressor),
])
def test_estimator_status(estimator):
    check_estimator(estimator)


@pytest.mark.parametrize(
    "n_rows,n_columns,unsuper_som,super_som, datapoint,expected", [
        (2, 2, np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
               np.array([[[0], [0.5]],
               [[1], [2]]]), np.array([0., 1.1, 2.1]).reshape(3,), np.array([0.])
        ),
])
def test_calc_estimation_output(n_rows, n_columns, unsuper_som, super_som, datapoint, expected):
    som = susi.SOMRegressor(n_rows=n_rows, n_columns=n_columns)
    som.unsuper_som_ = unsuper_som
    som.super_som_ = super_som
    assert np.array_equal(som.calc_estimation_output(datapoint), expected)
