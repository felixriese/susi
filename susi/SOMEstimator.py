"""SOMEstimator class."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm

from .SOMClustering import SOMClustering
from .SOMUtils import check_estimation_input, modify_weight_matrix_online


class SOMEstimator(SOMClustering, BaseEstimator, ABC):
    """Basic class for supervised self-organizing maps.

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
    node_list_ : np.ndarray of (int, int) tuples
        List of 2-dimensional coordinates of SOM nodes

    radius_max_ : float, int
        Maximum radius of the neighborhood function

    radius_min_ : float, int
        Minimum radius of the neighborhood function

    unsuper_som_ : np.ndarray
        Weight vectors of the unsupervised SOM
        shape = (self.n_rows, self.n_columns, X.shape[1])

    X_ : np.ndarray
        Input data

    fitted_ : bool
        States if estimator is fitted to X

    max_iterations_ : int
        Maximum number of iterations for the current training

    bmus_ :  list of (int, int) tuples
        List of best matching units (BMUs) of the dataset X

    sample_weights_ : np.ndarray
        Sample weights.

    n_features_in_ : int
        Number of input features

    """

    def __init__(
        self,
        n_rows: int = 10,
        n_columns: int = 10,
        *,
        init_mode_unsupervised: str = "random",
        init_mode_supervised: str = "random",
        n_iter_unsupervised: int = 1000,
        n_iter_supervised: int = 1000,
        train_mode_unsupervised: str = "online",
        train_mode_supervised: str = "online",
        neighborhood_mode_unsupervised: str = "linear",
        neighborhood_mode_supervised: str = "linear",
        learn_mode_unsupervised: str = "min",
        learn_mode_supervised: str = "min",
        distance_metric: str = "euclidean",
        learning_rate_start: float = 0.5,
        learning_rate_end: float = 0.05,
        nbh_dist_weight_mode: str = "pseudo-gaussian",
        missing_label_placeholder: Optional[Union[int, str]] = None,
        n_jobs: Optional[int] = None,
        random_state=None,
        verbose: Optional[int] = 0,
    ) -> None:
        """Initialize SOMEstimator object."""
        super().__init__(
            n_rows=n_rows,
            n_columns=n_columns,
            init_mode_unsupervised=init_mode_unsupervised,
            n_iter_unsupervised=n_iter_unsupervised,
            train_mode_unsupervised=train_mode_unsupervised,
            neighborhood_mode_unsupervised=neighborhood_mode_unsupervised,
            learn_mode_unsupervised=learn_mode_unsupervised,
            distance_metric=distance_metric,
            learning_rate_start=learning_rate_start,
            learning_rate_end=learning_rate_end,
            nbh_dist_weight_mode=nbh_dist_weight_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.init_mode_supervised = init_mode_supervised
        self.n_iter_supervised = n_iter_supervised
        self.train_mode_supervised = train_mode_supervised
        self.neighborhood_mode_supervised = neighborhood_mode_supervised
        self.learn_mode_supervised = learn_mode_supervised
        self.missing_label_placeholder = missing_label_placeholder

    @abstractmethod
    def _init_super_som(self) -> None:
        """Initialize map."""
        return None

    def fit(self, X: Sequence, y: Optional[Sequence] = None):
        """Fit supervised SOM to the input data.

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
        >>> som = susi.SOMRegressor()
        >>> som.fit(X, y)

        """
        X, y = check_estimation_input(X, y)
        self.X_: np.ndarray = X
        self.y_: np.ndarray = y
        self.n_features_in_ = self.X_.shape[1]

        return self._fit_estimator()

    def _fit_estimator(self):
        """Fit supervised SOM to the (checked) input data.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The prediction input samples.
        y : array-like matrix of shape = [n_samples, 1]
            The labels (ground truth) of the input samples

        """
        np.random.seed(seed=self.random_state)

        # supervised case:
        if self.missing_label_placeholder is None:
            self.labeled_indices_ = list(range(len(self.y_)))
            self.sample_weights_ = np.full(
                fill_value=1.0, shape=(len(self.X_), 1)
            )

        # semi-supervised case:
        else:
            self.labeled_indices_ = np.where(
                self.y_ != self.missing_label_placeholder
            )[0]
            unlabeled_weight = max(
                len(self.labeled_indices_) / len(self.y_), 0.1
            )
            self.sample_weights_ = np.full(
                fill_value=unlabeled_weight, shape=(len(self.X_), 1)
            )
            self.sample_weights_[self.labeled_indices_] = 1.0

        # train SOMs
        self._train_unsupervised_som()
        self._train_supervised_som()

        self.fitted_ = True

        return self

    def predict(self, X: Sequence, y: Optional[Sequence] = None) -> np.ndarray:
        """Predict output of data X.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The prediction input samples.
        y : None, optional
            Ignored.

        Returns
        -------
        y_pred : list of float
            List of predicted values.

        Examples
        --------
        Fit the SOM on your data `X, y`:

        >>> import susi
        >>> som = susi.SOMClassifier()
        >>> som.fit(X, y)
        >>> y_pred = som.predict(X)

        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X, dtype=np.float64)
        y_pred_list = []
        for dp in tqdm(X, desc="predict", **self.tqdm_params_):
            y_pred_list.append(self._calc_estimation_output(dp, proba=False))
        y_pred = np.array(y_pred_list)
        return y_pred

    def _calc_estimation_output(
        self, datapoint: np.ndarray, proba: bool = False
    ) -> Tuple[Union[int, str, float], np.ndarray]:
        """Get SOM output for fixed SOM.

        The given datapoint doesn't have to belong to the training set of the
        input SOM.

        Parameters
        ----------
        datapoint : np.ndarray, shape=(X.shape[1])
            Datapoint = one row of the dataset X
        proba : bool
            If True, probabilities are calculated.

        Returns
        -------
        int or str or float
            Content of SOM node which is linked to the datapoint.
            Classification: the label.
            Regression: the target variable.

        TODO Implement handling of incomplete datapoints

        """
        bmu_pos = self.get_bmu(datapoint, self.unsuper_som_)
        estimation_output = self.super_som_[bmu_pos[0], bmu_pos[1]][0]

        if not proba:
            return estimation_output

        return (estimation_output, self._calc_proba(bmu_pos=bmu_pos))

    def _calc_proba(self, bmu_pos: Tuple[int, int]) -> np.ndarray:
        """Calculate probabilities for datapoint related to BMU.

        .. versionadded:: 1.1.3

        This function is just a placeholder and should not be used.

        Parameters
        ----------
        bmu_pos : Tuple[int, int]
            BMU position on the SOM grid.

        Returns
        -------
        np.ndarray
            Dummy output.

        """
        return np.array([1.0])

    def _modify_weight_matrix_supervised(
        self,
        dist_weight_matrix: np.ndarray,
        true_vector: Optional[np.ndarray] = None,
        learning_rate: Optional[float] = None,
    ) -> np.ndarray:
        """Modify weights of the supervised SOM, either online or batch.

        Parameters
        ----------
        dist_weight_matrix : np.ndarray of float
            Current distance weight of the SOM for the specific node
        true_vector : np.ndarray, optional (default=None)
            True vector. `None` is only valid in batch mode.
        learning_rate : float, optional (default=None)
            Current learning rate of the SOM.  `None` is only valid in batch
            mode.

        Returns
        -------
        np.array
            Weight vector of the SOM after the modification

        """
        if self.train_mode_supervised == "online":
            # require valid values for true_vector and learning_rate
            if not isinstance(true_vector, np.ndarray) or not isinstance(
                learning_rate, float
            ):
                raise ValueError("Parameters required to be not None.")

            return modify_weight_matrix_online(
                som_array=self.super_som_,
                dist_weight_matrix=dist_weight_matrix,
                true_vector=true_vector,
                learning_rate=learning_rate,
            )

        if self.train_mode_supervised == "batch":
            return self._modify_weight_matrix_batch(
                som_array=self.super_som_,
                dist_weight_matrix=dist_weight_matrix[self.labeled_indices_],
                data=self.y_[self.labeled_indices_],
            )

        raise ValueError(
            "Invalid train_mode_supervised: " + str(self.train_mode_supervised)
        )

    def _train_supervised_som(self):
        """Train supervised SOM."""
        self._set_bmus(self.X_[self.labeled_indices_])
        self._init_super_som()

        if self.train_mode_supervised == "online":
            for it in tqdm(
                range(self.n_iter_supervised),
                desc="super",
                **self.tqdm_params_,
            ):
                # select one input vector & calculate best matching unit (BMU)
                dp_index = self._get_random_datapoint_index()
                bmu_pos = self.bmus_[dp_index]

                # calculate learning rate and neighborhood function
                learning_rate = self._calc_learning_rate(
                    curr_it=it, mode=self.learn_mode_supervised
                )
                nbh_func = self._calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_supervised
                )

                # calculate distance weight matrix and update weights
                dist_weight_matrix = self._get_nbh_distance_weight_matrix(
                    nbh_func, bmu_pos
                )
                self.super_som_ = self._modify_weight_matrix_supervised(
                    dist_weight_matrix=dist_weight_matrix,
                    true_vector=self.y_[self.labeled_indices_][dp_index],
                    learning_rate=learning_rate,
                )

        elif self.train_mode_supervised == "batch":
            for it in tqdm(
                range(self.n_iter_supervised),
                desc="super",
                **self.tqdm_params_,
            ):
                # calculate BMUs with the unsupervised (!) SOM
                bmus = self.get_bmus(self.X_)

                # calculate neighborhood function
                nbh_func = self._calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_supervised
                )

                # calculate distance weight matrix for all datapoints
                dist_weight_block = self._get_nbh_distance_weight_block(
                    nbh_func, bmus
                )

                self.super_som_ = self._modify_weight_matrix_supervised(
                    dist_weight_matrix=dist_weight_block
                )

    def fit_transform(
        self, X: Sequence, y: Optional[Sequence] = None
    ) -> np.ndarray:
        """Fit to the input data and transform it.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The training and prediction input samples.
        y : array-like matrix of shape = [n_samples, 1]
            The labels (ground truth) of the input samples

        Returns
        -------
        np.array of tuples (int, int)
            Predictions including the BMUs of each datapoint

        Examples
        --------
        Load the SOM, fit it to your input data `X` and transform your input
        data with:

        >>> import susi
        >>> som = susi.SOMClassifier()
        >>> tuples = som.fit_transform(X, y)

        """
        self.fit(X, y)
        self.X_ = check_array(X, dtype=np.float64)
        return self.transform(X, y)

    def get_estimation_map(self) -> np.ndarray:
        """Return SOM grid with the estimated value on each node.

        Returns
        -------
        super_som_ : np.ndarray
            Supervised SOM grid with estimated value on each node.

        Examples
        --------
        Fit the SOM on your data `X, y`:

        >>> import susi
        >>> import matplotlib.pyplot as plt
        >>> som = susi.SOMClassifier()
        >>> som.fit(X, y)
        >>> estimation_map = som.get_estimation_map()
        >>> plt.imshow(np.squeeze(estimation_map,) cmap="viridis_r")

        """
        return self.super_som_

    def _get_random_datapoint_index(self) -> int:
        """Find and return random datapoint index from labeled dataset.

        Returns
        -------
        int
            Random datapoint index from labeled dataset

        """
        if self.missing_label_placeholder is not None:
            random_datapoint: int = np.random.choice(
                len(self.y_[self.labeled_indices_])
            )
        else:
            random_datapoint = np.random.randint(low=0, high=len(self.y_))
        return random_datapoint

    def _more_tags(self) -> dict:
        """Add tags for `sklearn.utils.estimator_checks.check_estimator()`.

        Source
        ------
        https://scikit-learn.org/stable/developers/develop.html#estimator-tags

        """
        return {"preserves_dtype": []}
