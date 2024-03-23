"""SOMUtils functions."""

from typing import Sequence, Tuple

import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_X_y


def decreasing_rate(
    a_1: float, a_2: float, *, iteration_max: int, iteration: int, mode: str
) -> float:
    """Return a decreasing rate from collection.

    Parameters
    ----------
    a_1 : float
        Starting value of decreasing rate
    a_2 : float
        End value of decreasing rate
    iteration_max : int
        Maximum number of iterations
    iteration : int
        Current number of iterations
    mode : str
        Mode (= formula) of the decreasing rate

    Returns
    -------
    rate : float
        Decreasing rate

    Examples
    ---------
    >>> import susi
    >>> susi.decreasing_rate(0.8, 0.1, 100, 5, "exp")

    """
    rate = None
    if mode == "min":
        rate = a_1 * np.power(a_2 / a_1, iteration / iteration_max)

    elif mode == "exp":
        rate = a_1 * np.exp(-5 * np.divide(iteration, iteration_max))

    elif mode == "expsquare":
        rate = a_1 * np.exp(
            -5 * np.power(np.divide(iteration, iteration_max), 2)
        )

    elif mode == "linear":
        rate = (a_1 - a_2) * (1 - np.divide(iteration, iteration_max)) + a_2

    elif mode == "inverse":
        rate = a_1 / iteration

    elif mode == "root":
        rate = np.power(a_1, iteration / iteration_max)

    else:
        raise ValueError("Invalid decreasing rate mode: " + str(mode))

    # # prevent zero:
    # rate = max(rate, a_2)

    return rate


def check_estimation_input(
    X: Sequence, y: Sequence, *, is_classification: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Check input arrays.

    This function is adapted from sklearn.utils.validation.

    Parameters
    ----------
    X : nd-array or list
        Input data.
    y : nd-array, list
        Labels.
    is_classification : boolean (default=`False`)
        Wether the data is used for classification or regression tasks.

    Returns
    -------
    X : object
        The converted and validated `X`.
    y : object
        The converted and validated `y`.

    """
    if is_classification:
        X, y = check_X_y(X, y)
    else:
        X, y = check_X_y(X, y, dtype=np.float64)

    # TODO accept_sparse="csc"
    X = check_array(X, ensure_2d=True, dtype=np.float64)
    y = check_array(y, ensure_2d=False, dtype=None)

    if is_classification:
        check_classification_targets(y)

    y = np.atleast_1d(y)

    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    return X, y


def modify_weight_matrix_online(
    som_array: np.ndarray,
    *,
    dist_weight_matrix: np.ndarray,
    true_vector: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    """Modify weight matrix of the SOM for the online algorithm.

    Parameters
    ----------
    som_array : np.ndarray
        Weight vectors of the SOM
        shape = (self.n_rows, self.n_columns, X.shape[1])
    dist_weight_matrix : np.ndarray of float
        Current distance weight of the SOM for the specific node
    true_vector : np.ndarray
        True vector
    learning_rate : float
        Current learning rate of the SOM

    Returns
    -------
    np.array
        Weight vector of the SOM after the modification

    """
    return som_array + np.multiply(
        learning_rate,
        np.multiply(dist_weight_matrix, -np.subtract(som_array, true_vector)),
    )
