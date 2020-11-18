"""SOMRegressor class.

Copyright (c) 2019-2020, Felix M. Riese.
All rights reserved.

"""

import numpy as np
from sklearn.base import RegressorMixin

from .SOMEstimator import SOMEstimator


class SOMRegressor(SOMEstimator, RegressorMixin):
    """Supervised SOM for estimating continuous variables (= regression).

    Parameters
    ----------
    n_rows : int, optional (default=10)
        Number of rows for the SOM grid

    n_columns : int, optional (default=10)
        Number of columns for the SOM grid

    init_mode_unsupervised : str, optional (default="random")
        Initialization mode of the unsupervised SOM

    init_mode_supervised : str, optional (default="random")
        Initialization mode of the supervised SOM

    n_iter_unsupervised : int, optional (default=1000)
        Number of iterations for the unsupervised SOM

    n_iter_supervised : int, optional (default=1000)
        Number of iterations for the supervised SOM

    train_mode_unsupervised : str, optional (default="online")
        Training mode of the unsupervised SOM

    train_mode_supervised : str, optional (default="online")
        Training mode of the supervised SOM

    neighborhood_mode_unsupervised : str, optional (default="linear")
        Neighborhood mode of the unsupervised SOM

    neighborhood_mode_supervised : str, optional (default="linear")
        Neighborhood mode of the supervised SOM

    learn_mode_unsupervised : str, optional (default="min")
        Learning mode of the unsupervised SOM

    learn_mode_supervised : str, optional (default="min")
        Learning mode of the supervised SOM

    distance_metric : str, optional (default="euclidean")
        Distance metric to compare on feature level (not SOM grid).
        Possible metrics: {"euclidean", "manhattan", "mahalanobis",
        "tanimoto", "spectralangle"}. Note that "tanimoto" tends to be slow.

        .. versionadded:: 1.1.1
            Spectral angle metric.

    learning_rate_start : float, optional (default=0.5)
        Learning rate start value

    learning_rate_end : float, optional (default=0.05)
        Learning rate end value (only needed for some lr definitions)

    nbh_dist_weight_mode : str, optional (default="pseudo-gaussian")
        Formula of the neighborhood distance weight. Possible formulas
        are: {"pseudo-gaussian", "mexican-hat"}.

    missing_label_placeholder : int or str or None, optional (default=None)
        Label placeholder for datapoints with no label. This is needed for
        semi-supervised learning.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity.

    Attributes
    ----------
    node_list_ : np.array of (int, int) tuples
        List of 2-dimensional coordinates of SOM nodes

    radius_max_ : float, int
        Maximum radius of the neighborhood function

    radius_min_ : float, int
        Minimum radius of the neighborhood function

    unsuper_som_ : np.array
        Weight vectors of the unsupervised SOM
        shape = (self.n_rows, self.n_columns, X.shape[1])

    X_ : np.array
        Input data

    fitted_ : bool
        States if estimator is fitted to X

    max_iterations_ : int
        Maximum number of iterations for the current training

    bmus_ :  list of (int, int) tuples
        List of best matching units (BMUs) of the dataset X

    sample_weights_ : TODO

    n_regression_vars_ : int
        Number of regression variables. In most examples, this equals one.

    n_features_in_ : int
        Number of input features

    """

    def _init_super_som(self):
        """Initialize map for regression."""
        self.max_iterations_ = self.n_iter_supervised
        self.n_regression_vars_ = None

        # check if target variable has dimension 1 or >1
        if len(self.y_.shape) == 1:
            self.n_regression_vars_ = 1
        else:
            self.n_regression_vars_ = self.y_.shape[1]

        # initialize regression SOM
        if self.init_mode_supervised == "random":
            som = np.random.rand(self.n_rows, self.n_columns,
                                 self.n_regression_vars_)

        elif self.init_mode_supervised == "random_data":
            indices = np.random.randint(
                low=0, high=self.y_[self.labeled_indices_].shape[0],
                size=self.n_rows*self.n_columns)
            som_list = self.y_[self.labeled_indices_][indices]
            som = som_list.reshape(
                self.n_rows, self.n_columns, self.y_.shape[1])

        elif self.init_mode_supervised == "random_minmax":
            som = np.random.uniform(
                low=np.min(self.y_[self.labeled_indices_]),
                high=np.max(self.y_[self.labeled_indices_]),
                size=(self.n_rows, self.n_columns, self.n_regression_vars_))

        else:
            raise ValueError("Invalid init_mode_supervised: "+str(
                self.init_mode_supervised))

        self.super_som_ = som
