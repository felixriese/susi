"""SOMClassifier class.

Copyright (c) 2019-2020, Felix M. Riese.
All rights reserved.

"""

import numpy as np
from scipy.special import softmax
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight

from .SOMEstimator import SOMEstimator
from .SOMUtils import check_estimation_input


class SOMClassifier(SOMEstimator, ClassifierMixin):
    """Supervised SOM for estimating discrete variables (= classification).

    Parameters
    ----------
    n_rows : int, optional (default=10)
        Number of rows for the SOM grid

    n_columns : int, optional (default=10)
        Number of columns for the SOM grid

    init_mode_unsupervised : str, optional (default="random")
        Initialization mode of the unsupervised SOM

    init_mode_supervised : str, optional (default="majority")
        Initialization mode of the classification SOM

    n_iter_unsupervised : int, optional (default=1000)
        Number of iterations for the unsupervised SOM

    n_iter_supervised : int, optional (default=1000)
        Number of iterations for the classification SOM

    train_mode_unsupervised : str, optional (default="online")
        Training mode of the unsupervised SOM

    train_mode_supervised : str, optional (default="online")
        Training mode of the classification SOM

    neighborhood_mode_unsupervised : str, optional (default="linear")
        Neighborhood mode of the unsupervised SOM

    neighborhood_mode_supervised : str, optional (default="linear")
        Neighborhood mode of the classification SOM

    learn_mode_unsupervised : str, optional (default="min")
        Learning mode of the unsupervised SOM

    learn_mode_supervised : str, optional (default="min")
        Learning mode of the classification SOM

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

    do_class_weighting : bool, optional (default=True)
        If true, classes are weighted.

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

    placeholder_dict_ : dict
        Dict of placeholders for initializing nodes without mapped class.

    n_features_in_ : int
        Number of input features

    """

    def __init__(self,
                 n_rows: int = 10,
                 n_columns: int = 10, *,
                 init_mode_unsupervised: str = "random",
                 init_mode_supervised: str = "majority",
                 n_iter_unsupervised: int = 1000,
                 n_iter_supervised: int = 1000,
                 train_mode_unsupervised: str = "online",
                 train_mode_supervised: str = "online",
                 neighborhood_mode_unsupervised: str = "linear",
                 neighborhood_mode_supervised: str = "linear",
                 learn_mode_unsupervised: str = "min",
                 learn_mode_supervised: str = "min",
                 distance_metric: str = "euclidean",
                 learning_rate_start=0.5,
                 learning_rate_end=0.05,
                 nbh_dist_weight_mode: str = "pseudo-gaussian",
                 missing_label_placeholder=None,
                 do_class_weighting=True,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        """Initialize SOMClassifier object."""
        super().__init__(
            n_rows=n_rows,
            n_columns=n_columns,
            init_mode_unsupervised=init_mode_unsupervised,
            init_mode_supervised=init_mode_supervised,
            n_iter_unsupervised=n_iter_unsupervised,
            n_iter_supervised=n_iter_supervised,
            train_mode_unsupervised=train_mode_unsupervised,
            train_mode_supervised=train_mode_supervised,
            neighborhood_mode_unsupervised=neighborhood_mode_unsupervised,
            neighborhood_mode_supervised=neighborhood_mode_supervised,
            learn_mode_unsupervised=learn_mode_unsupervised,
            learn_mode_supervised=learn_mode_supervised,
            distance_metric=distance_metric,
            learning_rate_start=learning_rate_start,
            learning_rate_end=learning_rate_end,
            nbh_dist_weight_mode=nbh_dist_weight_mode,
            missing_label_placeholder=missing_label_placeholder,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.do_class_weighting = do_class_weighting

    def _init_super_som(self):
        """Initialize map."""
        self.max_iterations_ = self.n_iter_supervised

        self.placeholder_dict_ = {
            "str": "PLACEHOLDER",
            "int": -999999,
            "float": -99.999,
        }

        # get class information
        self.classes_, self.class_counts_ = np.unique(
            self.y_[self.labeled_indices_], return_counts=True)
        self.class_dtype_ = type(self.y_.flatten()[0])
        self._set_placeholder()

        # check if forbidden class name exists in classes
        if self.placeholder_ in self.classes_:
            raise ValueError("Forbidden class:", self.placeholder_)
        if self.placeholder_ == self.missing_label_placeholder:
            raise ValueError("Forbidden missing_label_placeholder:",
                             self.missing_label_placeholder)

        # class weighting:
        if self.do_class_weighting:
            self.class_weights_ = class_weight.compute_class_weight(
                'balanced', classes=np.unique(self.y_[self.labeled_indices_]),
                y=self.y_[self.labeled_indices_].flatten())
        else:
            self.class_weights_ = np.ones(shape=self.classes_.shape)

        # initialize classification SOM
        if self.init_mode_supervised == "majority":

            # define dtype
            if self.class_dtype_ in [str, np.str_]:
                init_dtype = "U" + str(len(max(np.unique(
                    self.y_[self.labeled_indices_]), key=len)))
            else:
                init_dtype = self.class_dtype_
            som = np.empty((self.n_rows, self.n_columns, 1),
                           dtype=init_dtype)

            for node in self.node_list_:
                dp_in_node = self.get_datapoints_from_node(node)

                # if no datapoint with label is mapped on this node:
                # node_class = self.placeholder_
                node_class = np.random.choice(
                    self.classes_,
                    p=self.class_counts_/np.sum(self.class_counts_))

                # if at least one datapoint with label is mapped to this node:
                if dp_in_node != []:
                    y_in_node = self.y_.flatten()[dp_in_node]
                    if not any(y_in_node == self.missing_label_placeholder):
                        node_class = np.argmax(
                            np.unique(y_in_node, return_counts=True)[1])

                som[node[0], node[1], 0] = node_class
        else:
            raise ValueError("Invalid init_mode_supervised: "+str(
                self.init_mode_supervised))

        self.super_som_ = som

    def _set_placeholder(self):
        """Set placeholder depending on the class dtype."""
        if self.class_dtype_ in [str, np.str_]:
            self.placeholder_ = self.placeholder_dict_["str"]
        elif self.class_dtype_ in [int, np.uint8, np.int64]:
            self.placeholder_ = self.placeholder_dict_["int"]
        elif self.class_dtype_ in [float, np.float_, np.float64]:
            self.placeholder_ = self.placeholder_dict_["float"]
        else:
            raise ValueError("No placeholder defined " +
                             "for the dtype of the classes:",
                             self.class_dtype_)

    def fit(self, X, y=None):
        """Fit classification SOM to the input data.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The prediction input samples.
        y : array-like matrix of shape = [n_samples, 1]
            The labels (ground truth) of the input samples

        Returns
        -------
        self : object

        Examples
        --------
        Load the SOM and fit it to your input data `X` and the labels `y` with:

        >>> import susi
        >>> som = susi.SOMClassifier()
        >>> som.fit(X, y)

        """
        X, y = check_estimation_input(X, y, is_classification=True)
        self.n_features_in_ = X.shape[1]

        return self._fit_estimator(X, y)

    def _modify_weight_matrix_supervised(self, dist_weight_matrix,
                                        true_vector=None,
                                        learningrate=None):
        """Modify weight matrix of the SOM.

        Parameters
        ----------
        dist_weight_matrix : np.array of float
            Current distance weight of the SOM for the specific node
        learningrate : float, optional
            Current learning rate of the SOM
        true_vector : np.array
            Datapoint = one row of the dataset X

        Returns
        -------
        new_matrix : np.array
            Weight vector of the SOM after the modification

        """
        new_matrix = None
        if self.train_mode_supervised == "online":
            class_weight = self.class_weights_[
                np.argwhere(self.classes_ == true_vector)[0, 0]]
            change_class_bool = self._change_class_proba(
                learningrate, dist_weight_matrix, class_weight)

            different_classes_matrix = (
                self.super_som_ != true_vector).reshape(
                    (self.n_rows, self.n_columns, 1))

            change_mask = np.multiply(change_class_bool,
                                      different_classes_matrix)

            new_matrix = np.copy(self.super_som_)
            new_matrix[change_mask] = true_vector

            new_matrix = new_matrix.reshape((self.n_rows, self.n_columns, 1))

        elif self.train_mode_supervised == "batch":
            # transform labels
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(self.y_)

            # calculate numerator and divisor for the batch formula
            numerator = np.sum(
                [np.multiply(y_bin[i], dist_weight_matrix[i].reshape(
                    (self.n_rows, self.n_columns, 1)))
                    for i in self.labeled_indices_], axis=0)

            # update weights
            new_matrix = lb.inverse_transform(
                softmax(numerator, axis=2).reshape(
                    (self.n_rows*self.n_columns, y_bin.shape[1]))).reshape(
                    (self.n_rows, self.n_columns, 1))

        else:
            raise ValueError("Invalid train_mode_supervised: "+str(
                self.train_mode_supervised))

        return new_matrix

    def _change_class_proba(self, learningrate, dist_weight_matrix,
                           class_weight):
        """Calculate probability of changing class in a node.

        Parameters
        ----------
        learningrate : float
            Current learning rate of the SOM
        dist_weight_matrix : np.array of float
            Current distance weight of the SOM for the specific node
        class_weight : float
            Weight of the class of the current datapoint

        Returns
        -------
        change_class_bool : np.array, shape = (n_rows, n_columns)
            Matrix with one boolean for each node on the SOM node.
            If true, the value of the respective SOM node gets changed.
            If false, the value of the respective SOM node stays the same.

        """
        _change_class_proba = learningrate * dist_weight_matrix
        _change_class_proba *= class_weight
        random_matrix = np.random.rand(self.n_rows, self.n_columns, 1)
        change_class_bool = random_matrix < _change_class_proba
        return change_class_bool
