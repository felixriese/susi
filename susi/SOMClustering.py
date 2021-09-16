"""SOMClustering class.

Copyright (c) 2019-2021 Felix M. Riese.
All rights reserved.

"""

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.spatial.distance as dist
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import binarize
from sklearn.utils.validation import check_array
from tqdm import tqdm

from .SOMUtils import decreasing_rate, modify_weight_matrix_online


class SOMClustering:
    """Unsupervised self-organizing map for clustering.

    Parameters
    ----------
    n_rows : int, optional (default=10)
        Number of rows for the SOM grid

    n_columns : int, optional (default=10)
        Number of columns for the SOM grid

    init_mode_unsupervised : str, optional (default="random")
        Initialization mode of the unsupervised SOM

    n_iter_unsupervised : int, optional (default=1000)
        Number of iterations for the unsupervised SOM

    train_mode_unsupervised : str, optional (default="online")
        Training mode of the unsupervised SOM

    neighborhood_mode_unsupervised : str, optional (default="linear")
        Neighborhood mode of the unsupervised SOM

    learn_mode_unsupervised : str, optional (default="min")
        Learning mode of the unsupervised SOM

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

    fitted_ : boolean
        States if estimator is fitted to X

    max_iterations_ : int
        Maximum number of iterations for the current training

    bmus_ :  list of (int, int) tuples
        List of best matching units (BMUs) of the dataset X

    variances_ : array of float
        Standard deviations of every feature

    """

    def __init__(
        self,
        n_rows: int = 10,
        n_columns: int = 10,
        *,
        init_mode_unsupervised: str = "random",
        n_iter_unsupervised: int = 1000,
        train_mode_unsupervised: str = "online",
        neighborhood_mode_unsupervised: str = "linear",
        learn_mode_unsupervised: str = "min",
        distance_metric: str = "euclidean",
        learning_rate_start: float = 0.5,
        learning_rate_end: float = 0.05,
        nbh_dist_weight_mode: str = "pseudo-gaussian",
        n_jobs: Optional[int] = None,
        random_state=None,
        verbose: Optional[int] = 0,
    ) -> None:
        """Initialize SOMClustering object."""
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.init_mode_unsupervised = init_mode_unsupervised
        self.n_iter_unsupervised = n_iter_unsupervised
        self.train_mode_unsupervised = train_mode_unsupervised
        self.neighborhood_mode_unsupervised = neighborhood_mode_unsupervised
        self.learn_mode_unsupervised = learn_mode_unsupervised
        self.distance_metric = distance_metric
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.nbh_dist_weight_mode = nbh_dist_weight_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _init_unsuper_som(self) -> None:
        """Initialize map."""
        # init node list
        self.node_list_ = np.array(
            list(itertools.product(range(self.n_rows), range(self.n_columns))),
            dtype=int,
        )

        self.max_iterations_ = self.n_iter_unsupervised

        # init radius parameter
        self.radius_max_ = max(self.n_rows, self.n_columns) / 2
        self.radius_min_ = 1

        # tqdm paramters
        self.tqdm_params_ = {"disable": not bool(self.verbose), "ncols": 100}

        # init unsupervised SOM in the feature space
        if self.init_mode_unsupervised == "random":
            som = np.random.rand(self.n_rows, self.n_columns, self.X_.shape[1])

        elif self.init_mode_unsupervised == "random_data":
            indices = np.random.randint(
                low=0, high=self.X_.shape[0], size=self.n_rows * self.n_columns
            )
            som_list = self.X_[indices]
            som = som_list.reshape(
                self.n_rows, self.n_columns, self.X_.shape[1]
            )

        elif self.init_mode_unsupervised == "pca":

            # fixed number of components
            pca = PCA(n_components=2, random_state=self.random_state)

            pca_comp = pca.fit(self.X_).components_

            a_row = np.linspace(-1.0, 1.0, self.n_rows)
            a_col = np.linspace(-1.0, 1.0, self.n_columns)

            som = np.zeros(
                shape=(self.n_rows, self.n_columns, self.X_.shape[1])
            )

            for node in self.node_list_:
                som[node[0], node[1], :] = np.add(
                    np.multiply(a_row[node[0]], pca_comp[0]),
                    np.multiply(a_col[node[1]], pca_comp[1]),
                )

        else:
            raise ValueError(
                f"Invalid init_mode_unsupervised: {self.init_mode_unsupervised}."
            )

        self.unsuper_som_ = som

    def fit(self, X: Sequence, y: Optional[Sequence] = None):
        """Fit unsupervised SOM to input data.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            Not used in this class.

        Returns
        -------
        self : object

        Examples
        --------
        Load the SOM and fit it to your input data `X` with:

        >>> import susi
        >>> som = susi.SOMClustering()
        >>> som.fit(X)

        """
        np.random.seed(seed=self.random_state)
        self.X_ = check_array(X, dtype=np.float64)  # TODO accept_sparse

        self.sample_weights_ = np.full(fill_value=1.0, shape=(len(self.X_), 1))

        self._train_unsupervised_som()
        self.fitted_ = True

        return self

    def _train_unsupervised_som(self) -> None:
        """Train unsupervised SOM."""
        self._init_unsuper_som()

        if self.train_mode_unsupervised == "online":
            for it in tqdm(
                range(self.n_iter_unsupervised),
                desc="unsuper",
                **self.tqdm_params_,
            ):

                # select one input vector & calculate best matching unit (BMU)
                dp = np.random.randint(low=0, high=len(self.X_))
                bmu_pos = self.get_bmu(self.X_[dp], self.unsuper_som_)

                # calculate learning rate and neighborhood function
                learning_rate = self._calc_learning_rate(
                    curr_it=it, mode=self.learn_mode_unsupervised
                )
                nbh_func = self._calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_unsupervised
                )

                # calculate distance weight matrix and update weights
                dist_weight_matrix = self._get_nbh_distance_weight_matrix(
                    nbh_func, bmu_pos
                )
                self.unsuper_som_ = modify_weight_matrix_online(
                    som_array=self.unsuper_som_,
                    dist_weight_matrix=dist_weight_matrix,
                    true_vector=self.X_[dp],
                    learning_rate=learning_rate * self.sample_weights_[dp],
                )

        elif self.train_mode_unsupervised == "batch":
            for it in tqdm(
                range(self.n_iter_unsupervised),
                desc="unsuper",
                **self.tqdm_params_,
            ):

                # calculate BMUs
                bmus = self.get_bmus(self.X_)

                # calculate neighborhood function
                nbh_func = self._calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_unsupervised
                )

                # calculate distance weight matrix for all datapoints
                dist_weight_block = self._get_nbh_distance_weight_block(
                    nbh_func, bmus
                )

                # update weights
                self.unsuper_som_ = self._modify_weight_matrix_batch(
                    self.unsuper_som_, dist_weight_block, self.X_
                )

        else:
            raise NotImplementedError(
                "Unsupervised mode not implemented:",
                self.train_mode_unsupervised,
            )

        self._set_bmus(self.X_)

    def _calc_learning_rate(self, curr_it: int, mode: str) -> float:
        """Calculate learning rate alpha with 0 <= alpha <= 1.

        Parameters
        ----------
        curr_it : int
            Current iteration count
        mode : str
            Mode of the learning rate (min, exp, expsquare)

        Returns
        -------
        float
            Learning rate

        """
        return decreasing_rate(
            self.learning_rate_start,
            self.learning_rate_end,
            iteration_max=self.max_iterations_,
            iteration=curr_it,
            mode=mode,
        )

    def _calc_neighborhood_func(self, curr_it: int, mode: str) -> float:
        """Calculate neighborhood function (= radius).

        Parameters
        ----------
        curr_it : int
            Current number of iterations
        mode : str
            Mode of the decreasing rate

        Returns
        -------
        float
            Neighborhood function (= radius)

        """
        return decreasing_rate(
            self.radius_max_,
            self.radius_min_,
            iteration_max=self.max_iterations_,
            iteration=curr_it,
            mode=mode,
        )

    def get_bmu(
        self, datapoint: np.ndarray, som_array: np.ndarray
    ) -> Tuple[int, int]:
        """Get best matching unit (BMU) for datapoint.

        Parameters
        ----------
        datapoint : np.ndarray, shape=shape[1]
            Datapoint = one row of the dataset X
        som_array : np.ndarray
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])

        Returns
        -------
        tuple, shape = (int, int)
            Position of best matching unit (row, column)

        """
        a = self._get_node_distance_matrix(
            datapoint.astype(np.float64), som_array
        )

        return np.argwhere(a == np.min(a))[0]

    def get_bmus(
        self, X: np.ndarray, som_array: Optional[np.array] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """Get Best Matching Units for big datalist.

        Parameters
        ----------
        X : np.ndarray
            List of datapoints
        som_array : np.ndarray, optional (default=`None`)
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])

        Returns
        -------
        bmus : list of (int, int) tuples
            Position of best matching units (row, column) for each datapoint

        Examples
        --------
        Load the SOM, fit it to your input data `X` and transform your input
        data with:

        >>> import susi
        >>> import matplotlib.pyplot as plt
        >>> som = susi.SOMClustering()
        >>> som.fit(X)
        >>> bmu_list = som.get_bmus(X)
        >>> plt.hist2d([x[0] for x in bmu_list], [x[1] for x in bmu_list]

        """
        if som_array is None:
            som_array = self.unsuper_som_

        bmus = None
        if self.n_jobs == 1:
            bmus = [tuple(self.get_bmu(dp, som_array)) for dp in X]
        else:
            n_jobs, _, _ = self._partition_bmus(X)
            bmus = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(self.get_bmu)(dp, som_array) for dp in X
            )
        return bmus

    def _partition_bmus(
        self, X: np.ndarray
    ) -> Tuple[float, List[int], List[int]]:
        """Private function used to partition bmus between jobs.

        Parameters
        ----------
        X : np.ndarray
            List of datapoints

        Returns
        -------
        n_jobs : int
            Number of jobs
        list of int
            List of number of datapoints per job
        list of int
            List of start values for every job list

        """
        n_datapoints = len(X)
        n_jobs = min(effective_n_jobs(self.n_jobs), n_datapoints)

        n_datapoints_per_job = np.full(
            n_jobs, n_datapoints // n_jobs, dtype=int
        )

        n_datapoints_per_job[: n_datapoints % n_jobs] += 1
        starts = np.cumsum(n_datapoints_per_job)

        return n_jobs, n_datapoints_per_job.tolist(), [0] + starts.tolist()

    def _set_bmus(
        self, X: np.ndarray, som_array: Optional[np.array] = None
    ) -> None:
        """Set BMUs in the current SOM object.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples.
        som_array : np.ndarray
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])

        """
        self.bmus_ = self.get_bmus(X=X, som_array=som_array)

    def _get_node_distance_matrix(
        self, datapoint: np.ndarray, som_array: np.ndarray
    ) -> np.ndarray:
        """Get distance of datapoint and node using Euclidean distance.

        Parameters
        ----------
        datapoint : np.ndarray, shape=(X.shape[1])
            Datapoint = one row of the dataset `X`
        som_array : np.ndarray
            Weight vectors of the SOM,
            shape = (self.n_rows, self.n_columns, X.shape[1])

        Returns
        -------
        distmat : np.ndarray of float
            Distance between datapoint and each SOM node

        """
        # algorithms on the full matrix
        if self.distance_metric == "euclidean":
            return np.linalg.norm(som_array - datapoint, axis=2)

        # node-by-node algorithms
        distmat = np.zeros((self.n_rows, self.n_columns))
        if self.distance_metric == "manhattan":
            for node in self.node_list_:
                distmat[node] = dist.cityblock(
                    som_array[node[0], node[1]], datapoint
                )

        elif self.distance_metric == "mahalanobis":
            for node in self.node_list_:
                som_node = som_array[node[0], node[1]]
                cov = np.cov(
                    np.stack((datapoint, som_node), axis=0), rowvar=False
                )
                cov_pinv = np.linalg.pinv(cov)  # pseudo-inverse
                distmat[node] = dist.mahalanobis(datapoint, som_node, cov_pinv)

        elif self.distance_metric == "tanimoto":
            # Note that this is a binary distance measure.
            # Therefore, the vectors have to be converted.
            # Source: Melssen 2006, Supervised Kohonen networks for
            #         classification problems
            # VERY SLOW ALGORITHM!!!
            threshold = 0.5
            for node in self.node_list_:
                som_node = som_array[node[0], node[1]]
                distmat[node] = dist.rogerstanimoto(
                    binarize(
                        datapoint.reshape(1, -1),
                        threshold=threshold,
                        copy=True,
                    ),
                    binarize(
                        som_node.reshape(1, -1), threshold=threshold, copy=True
                    ),
                )

        elif self.distance_metric == "spectralangle":
            for node in self.node_list_:
                distmat[node] = np.arccos(
                    np.divide(
                        np.dot(som_array[node[0], node[1]], datapoint),
                        np.multiply(
                            np.linalg.norm(som_array),
                            np.linalg.norm(datapoint),
                        ),
                    )
                )

        return distmat

    def _get_nbh_distance_weight_matrix(
        self, neighborhood_func: float, bmu_pos: Tuple[int, int]
    ) -> np.ndarray:
        """Calculate neighborhood distance weight.

        Parameters
        ----------
        neighborhood_func : float
            Current neighborhood function
        bmu_pos : tuple, shape=(int, int)
            Position of calculated BMU of the current datapoint

        Returns
        -------
        np.array of float, shape=(n_rows, n_columns)
            Neighborhood distance weight matrix between SOM and BMU

        """
        dist_mat = np.linalg.norm(self.node_list_ - bmu_pos, axis=1)

        pseudogaussian = np.exp(
            -np.divide(
                np.power(dist_mat, 2), (2 * np.power(neighborhood_func, 2))
            )
        )

        if self.nbh_dist_weight_mode == "pseudo-gaussian":
            return pseudogaussian.reshape((self.n_rows, self.n_columns, 1))

        if self.nbh_dist_weight_mode == "mexican-hat":
            mexicanhat = np.multiply(
                pseudogaussian,
                np.subtract(
                    1,
                    np.divide(
                        np.power(dist_mat, 2), np.power(neighborhood_func, 2)
                    ),
                ),
            )
            return mexicanhat.reshape((self.n_rows, self.n_columns, 1))

        raise ValueError(
            "Invalid nbh_dist_weight_mode: " + str(self.nbh_dist_weight_mode)
        )

    def _get_nbh_distance_weight_block(
        self, nbh_func: float, bmus: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Calculate distance weight matrix for all datapoints.

        The combination of several distance weight matrices is called
        "block" in the following.

        Parameters
        ----------
        neighborhood_func : float
            Current neighborhood function
        bmus : list of tuple (int, int)
            Positions of calculated BMUs of the datapoints

        Returns
        -------
        dist_weight_block : np.ndarray of float, shape=(n_rows, n_columns)
            Neighborhood distance weight block between SOM and BMUs

        """
        dist_weight_block = np.zeros((len(bmus), self.n_rows, self.n_columns))

        for i, bmu_pos in enumerate(bmus):
            dist_weight_block[i] = self._get_nbh_distance_weight_matrix(
                nbh_func, bmu_pos
            ).reshape((self.n_rows, self.n_columns))

        return dist_weight_block

    def _modify_weight_matrix_batch(
        self,
        som_array: np.ndarray,
        dist_weight_matrix: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        """Modify weight matrix of the SOM for the online algorithm.

        Parameters
        ----------
        som_array : np.ndarray
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])
        dist_weight_matrix : np.ndarray of float
            Current distance weight of the SOM for the specific node
        data : np.ndarray
            True vector(s)
        learning_rate : float
            Current learning rate of the SOM

        Returns
        -------
        np.array
            Weight vector of the SOM after the modification

        """
        # calculate numerator and divisor for the batch formula
        numerator = np.sum(
            [
                np.multiply(
                    data[i],
                    dist_weight_matrix[i].reshape(
                        (self.n_rows, self.n_columns, 1)
                    ),
                )
                for i in range(len(data))
            ],
            axis=0,
        )
        divisor = np.sum(dist_weight_matrix, axis=0).reshape(
            (self.n_rows, self.n_columns, 1)
        )

        # update weights
        old_som = np.copy(som_array)
        new_som = np.divide(
            numerator,
            divisor,
            out=np.full_like(numerator, np.nan),
            where=(divisor != 0),
        )

        # overwrite new nans with old entries
        new_som[np.isnan(new_som)] = old_som[np.isnan(new_som)]
        return new_som

    def transform(
        self, X: Sequence, y: Optional[Sequence] = None
    ) -> np.ndarray:
        """Transform input data.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The prediction input samples.
        y : None, optional
            Ignored.

        Returns
        -------
        np.array of tuples (int, int)
            Predictions including the BMUs of each datapoint

        Examples
        --------
        Load the SOM, fit it to your input data `X` and transform your input
        data with:

        >>> import susi
        >>> som = susi.SOMClustering()
        >>> som.fit(X)
        >>> X_transformed = som.transform(X)

        """
        # assert(self.fitted_ is True)
        self.X_ = check_array(X, dtype=np.float64)
        return np.array(self.get_bmus(self.X_))

    def fit_transform(
        self, X: Sequence, y: Optional[Sequence] = None
    ) -> np.ndarray:
        """Fit to the input data and transform it.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The training and prediction input samples.
        y : None, optional
            Ignored.

        Returns
        -------
        np.array of tuples (int, int)
            Predictions including the BMUs of each datapoint

        Examples
        --------
        Load the SOM, fit it to your input data `X` and transform your input
        data with:

        >>> import susi
        >>> som = susi.SOMClustering()
        >>> X_transformed = som.fit_transform(X)

        """
        self.fit(X)
        # assert(self.fitted_ is True)
        self.X_ = check_array(X, dtype=np.float64)
        return self.transform(X, y)

    def get_datapoints_from_node(self, node: Tuple[int, int]) -> List[int]:
        """Get all datapoints of one node.

        Parameters
        ----------
        node : tuple, shape (int, int)
            Node for which the linked datapoints are calculated

        Returns
        -------
        datapoints : list of int
            List of indices of the datapoints that are linked to `node`

        """
        datapoints = []
        for i in range(len(self.bmus_)):
            if np.array_equal(self.bmus_[i], node):
                datapoints.append(i)
        return datapoints

    def get_clusters(self, X: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Calculate the SOM nodes on the unsupervised SOM grid per datapoint.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        list of tuples (int, int)
            List of SOM nodes, one for each input datapoint

        """
        return self.get_bmus(X)

    def get_u_matrix(self, mode: str = "mean") -> np.ndarray:
        """Calculate unified distance matrix (u-matrix).

        Parameters
        ----------
        mode : str, optional (default="mean)
            Choice of the averaging algorithm

        Returns
        -------
        u_matrix : np.ndarray
            U-matrix containing the distances between all nodes of the
            unsupervised SOM. Shape = (n_rows*2-1, n_columns*2-1)

        Examples
        --------
        Fit your SOM to input data `X` and then calculate the u-matrix with
        `get_u_matrix()`. You can plot the u-matrix then with e.g.
        `pyplot.imshow()`.

        >>> import susi
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> som = susi.SOMClustering()
        >>> som.fit(X)
        >>> umat = som.get_u_matrix()
        >>> plt.imshow(np.squeeze(umat))

        """
        self.u_mean_mode_ = mode

        self.u_matrix = np.zeros(
            shape=(self.n_rows * 2 - 1, self.n_columns * 2 - 1, 1), dtype=float
        )

        # step 1: fill values between SOM nodes
        self._calc_u_matrix_distances()

        # step 2: fill values at SOM nodes and on diagonals
        self._calc_u_matrix_means()

        return self.u_matrix

    def _calc_u_matrix_distances(self) -> None:
        """Calculate the Eucl. distances between all neighbored SOM nodes."""
        for u_node in itertools.product(
            range(self.n_rows * 2 - 1), range(self.n_columns * 2 - 1)
        ):

            # neighbor vector
            nb = (0, 0)

            if not (u_node[0] % 2) and (u_node[1] % 2):
                # mean horizontally
                nb = (0, 1)

            elif (u_node[0] % 2) and not (u_node[1] % 2):
                # mean vertically
                nb = (1, 0)

            self.u_matrix[u_node] = np.linalg.norm(
                self.unsuper_som_[u_node[0] // 2][u_node[1] // 2]
                - self.unsuper_som_[u_node[0] // 2 + nb[0]][
                    u_node[1] // 2 + nb[1]
                ],
                axis=0,
            )

    def _calc_u_matrix_means(self) -> None:
        """Calculate the missing parts of the u-matrix.

        After `_calc_u_matrix_distances()`, there are two kinds of entries
        missing: the entries at the positions of the actual SOM nodes and the
        entries in between the distance nodes. Both types of entries are
        calculated in this function.

        """
        for u_node in itertools.product(
            range(self.n_rows * 2 - 1), range(self.n_columns * 2 - 1)
        ):

            if not (u_node[0] % 2) and not (u_node[1] % 2):
                # SOM nodes -> mean over 2-4 values

                nodelist = []
                if u_node[0] > 0:
                    nodelist.append((u_node[0] - 1, u_node[1]))
                if u_node[0] < self.n_rows * 2 - 2:
                    nodelist.append((u_node[0] + 1, u_node[1]))
                if u_node[1] > 0:
                    nodelist.append((u_node[0], u_node[1] - 1))
                if u_node[1] < self.n_columns * 2 - 2:
                    nodelist.append((u_node[0], u_node[1] + 1))
                self.u_matrix[u_node] = self._get_u_mean(nodelist)

            elif (u_node[0] % 2) and (u_node[1] % 2):
                # mean over four

                self.u_matrix[u_node] = self._get_u_mean(
                    [
                        (u_node[0] - 1, u_node[1]),
                        (u_node[0] + 1, u_node[1]),
                        (u_node[0], u_node[1] - 1),
                        (u_node[0], u_node[1] + 1),
                    ]
                )

    def _get_u_mean(self, nodelist: List[Tuple[int, int]]) -> Optional[float]:
        """Calculate a mean value of the node entries in `nodelist`.

        Parameters
        ----------
        nodelist : list of tuple (int, int)
            List of nodes on the u-matrix containing distance values

        Returns
        -------
        u_mean : float
            Mean value

        """
        meanlist = [self.u_matrix[u_node] for u_node in nodelist]
        u_mean = None
        if self.u_mean_mode_ == "mean":
            u_mean = np.mean(meanlist)
        elif self.u_mean_mode_ == "median":
            u_mean = np.median(meanlist)
        elif self.u_mean_mode_ == "min":
            u_mean = np.min(meanlist)
        elif self.u_mean_mode_ == "max":
            u_mean = np.max(meanlist)
        return u_mean

    def _get_node_neighbors(
        self, node: Tuple[int, int], radius: int = 1
    ) -> List[Tuple[int, int]]:
        """Get neighboring nodes (grid parameters) of `node`.

        .. versionadded:: 1.1.3

        Parameters
        ----------
        node : Tuple[int, int]
            Node position on the SOM grid.
        radius : int, optional (default=1)
            Radius to calculate the node radius. Is set arbitrarily to 1, can
            be changed in future versions.

        Returns
        -------
        [type]
            [description]
        """
        row_range = range(
            max(node[0] - radius, 0),
            min(node[0] + radius, self.n_rows - 1) + 1,
        )
        column_range = range(
            max(node[1] - radius, 0),
            min(node[1] + radius, self.n_columns - 1) + 1,
        )
        return list(itertools.product(row_range, column_range))

    def get_quantization_error(self, X: Optional[Sequence] = None) -> float:
        """Get quantization error for `X` (or the training data).

        Parameters
        ----------
        X : array-like matrix, optional (default=True)
            Samples of shape = [n_samples, n_features]. If `None`, the training
            data is used for the calculation.

        Returns
        -------
        float
            Mean quantization error over all datapoints.

        Raises
        ------
        RuntimeError
            Raised if the SOM is not fitted yet.

        """
        if not self.fitted_:
            raise RuntimeError("SOM is not fitted!")

        if X is None:
            X = self.X_

        weights_per_datapoint = [
            self.unsuper_som_[bmu[0], bmu[1]] for bmu in self.get_bmus(X)
        ]

        quantization_errors = np.linalg.norm(
            np.subtract(weights_per_datapoint, X)
        )

        return np.mean(quantization_errors)
