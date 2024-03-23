"""Test for susi.SOMRegressor.

Usage:
python -m pytest tests/test_SOMRegressor.py

"""

import itertools
import os
import sys

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import susi  # noqa


@pytest.fixture
def training_data():
    """Get training data for supervised regression."""
    # define test dataset
    cali = fetch_california_housing()
    n_datapoints: int = 100
    X_all = cali.data[:n_datapoints]
    y_all = cali.target[:n_datapoints]
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.5, random_state=42
    )

    # preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_orig)
    X_test = scaler.transform(X_test_orig)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def training_data_semi(training_data):
    """Get training data for semi-supervised regression."""
    X_train, X_test, y_train, y_test = training_data

    # data with missing labels -> semi-supervised
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_train)) < 0.5
    y_train_semi = np.copy(y_train)
    y_train_semi[random_unlabeled_points] = -1

    return X_train, X_test, y_train_semi, y_test


# SOM variables
TRAIN_MODES = ["online", "batch"]


@pytest.mark.parametrize(
    "n_rows,n_columns",
    [
        (10, 10),
        (12, 15),
    ],
)
def test_som_regressor_init(n_rows, n_columns):
    som_reg = susi.SOMRegressor(n_rows=n_rows, n_columns=n_columns)
    assert som_reg.n_rows == n_rows
    assert som_reg.n_columns == n_columns


@pytest.mark.parametrize(
    "X,y,init_mode",
    [
        (
            np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]),
            np.array([[3], [5]]),
            "random",
        ),
        (
            np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]),
            np.array([[3], [5]]),
            "random_data",
        ),
        (
            np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]),
            np.array([[3], [5]]),
            "random_minmax",
        ),
        (
            np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]),
            np.array([5, 5]),
            "random",
        ),
    ],
)
def test_init_super_som_regressor(X, y, init_mode):
    som = susi.SOMRegressor(init_mode_supervised=init_mode)
    som.X_ = X
    som.y_ = y
    som.labeled_indices_ = np.where(som.y_ != -1)[0]
    som._init_super_som()

    # test type
    assert isinstance(som.super_som_, np.ndarray)

    # test shape
    n_rows = som.n_rows
    n_columns = som.n_columns
    assert som.super_som_.shape == (n_rows, n_columns, som.n_regression_vars_)

    with pytest.raises(ValueError):
        som = susi.SOMRegressor(init_mode_supervised="wrong")
        som.X_ = X
        som.y_ = y
        som._init_super_som()


@pytest.mark.parametrize(
    "train_mode_unsupervised,train_mode_supervised",
    itertools.product(TRAIN_MODES, TRAIN_MODES),
)
def test_predict(
    training_data, train_mode_unsupervised, train_mode_supervised
):
    X_train, X_test, y_train, y_test = training_data

    som_reg = susi.SOMRegressor(
        n_rows=3,
        n_columns=3,
        train_mode_unsupervised=train_mode_unsupervised,
        train_mode_supervised=train_mode_supervised,
        n_iter_unsupervised=100,
        n_iter_supervised=100,
        random_state=42,
    )

    som_reg.fit(X_train, y_train)
    y_pred = som_reg.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_estimator_status():
    check_estimator(
        susi.SOMRegressor(
            n_iter_unsupervised=10000,
            n_iter_supervised=10000,
            n_rows=30,
            n_columns=30,
        )
    )


@pytest.mark.parametrize(
    "n_rows,n_columns,unsuper_som,super_som,datapoint,expected",
    [
        (
            2,
            2,
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array([[[0], [0.5]], [[1], [2]]]),
            np.array([0.0, 1.1, 2.1]).reshape(
                3,
            ),
            0.0,
        ),
    ],
)
def test_calc_estimation_output(
    n_rows, n_columns, unsuper_som, super_som, datapoint, expected
):
    som = susi.SOMRegressor(n_rows=n_rows, n_columns=n_columns)
    som.unsuper_som_ = unsuper_som
    som.super_som_ = super_som
    output = som._calc_estimation_output(datapoint)
    print(output)
    assert np.array_equal(output, expected)


def test_mexicanhat_nbh_dist_weight_mode(training_data):
    X_train, X_test, y_train, _ = training_data
    som = susi.SOMRegressor(nbh_dist_weight_mode="mexican-hat")
    som.fit(X_train, y_train)
    som.predict(X_test)
    with pytest.raises(ValueError):
        som = susi.SOMRegressor(nbh_dist_weight_mode="pseudogaussian")
        som.fit(X_train, y_train)


@pytest.mark.parametrize(
    "train_mode_unsupervised,train_mode_supervised",
    itertools.product(TRAIN_MODES, TRAIN_MODES),
)
def test_semisupervised_regressor(
    training_data_semi, train_mode_unsupervised, train_mode_supervised
):
    X_train, X_test, y_train_semi, y_test = training_data_semi
    som_reg = susi.SOMRegressor(
        n_rows=3,
        n_columns=3,
        train_mode_unsupervised=train_mode_unsupervised,
        train_mode_supervised=train_mode_supervised,
        n_iter_unsupervised=100,
        n_iter_supervised=100,
        missing_label_placeholder=-1,
        random_state=42,
    )

    som_reg.fit(X_train, y_train_semi)
    y_pred = som_reg.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_modify_weight_matrix_supervised():
    som = susi.SOMRegressor(train_mode_supervised="online")

    with pytest.raises(ValueError):
        som._modify_weight_matrix_supervised(
            dist_weight_matrix=np.array([1.0, 2.0]),
            true_vector=None,
            learning_rate=1.0,
        )
    with pytest.raises(ValueError):
        som._modify_weight_matrix_supervised(
            dist_weight_matrix=np.array([1.0, 2.0]),
            true_vector=np.array([1.0, 2.0]),
            learning_rate=None,
        )

    som = susi.SOMRegressor(train_mode_supervised="wrong")
    with pytest.raises(ValueError):
        som._modify_weight_matrix_supervised(
            dist_weight_matrix=np.array([1.0, 2.0]),
            true_vector=np.array([1.0, 2.0]),
            learning_rate=None,
        )
