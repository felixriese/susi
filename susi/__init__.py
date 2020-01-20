"""SuSi: Supervised Self-Organining Map.

Python package for unsupervised, supervised and semi-supervised
self-organizing maps (SOM).

"""

import itertools
from abc import ABC, abstractmethod

import numpy as np
import scipy.spatial.distance as dist
from joblib import Parallel, delayed, effective_n_jobs
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, binarize
from sklearn.utils import class_weight
from sklearn.utils.fixes import parallel_helper
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm import tqdm


def decreasing_rate(a_1, a_2, iteration_max, iteration, mode):
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
            -5 * np.power(np.divide(iteration, iteration_max), 2))

    elif mode == "linear":
        rate = (a_1-a_2)*(1 - np.divide(iteration, iteration_max)) + a_2

    elif mode == "inverse":
        rate = a_1 / iteration

    elif mode == "root":
        rate = np.power(a_1, iteration / iteration_max)

    else:
        raise ValueError("Invalid decreasing rate mode: "+str(mode))

    # # prevent zero:
    # rate = max(rate, a_2)

    return rate


def check_estimation_input(X, y, is_classification=False):
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


def modify_weight_matrix_online(som_array, dist_weight_matrix,
                                true_vector, learningrate):
    """Modify weight matrix of the SOM for the online algorithm.

    Parameters
    ----------
    som_array : np.array
        Weight vectors of the SOM
        shape = (self.n_rows, self.n_columns, X.shape[1])
    dist_weight_matrix : np.array of float
        Current distance weight of the SOM for the specific node
    true_vector : np.array
        True vector
    learningrate : float
        Current learning rate of the SOM

    Returns
    -------
    np.array
        Weight vector of the SOM after the modification

    """
    return som_array + np.multiply(learningrate, np.multiply(
        dist_weight_matrix, -np.subtract(som_array, true_vector)))


class SOMClustering():
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
        "tanimoto"}. Note that "tanimoto" tends to be slow.

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

    fitted_ : boolean
        States if estimator is fitted to X

    max_iterations_ : int
        Maximum number of iterations for the current training

    bmus_ :  list of (int, int) tuples
        List of best matching units (BMUs) of the dataset X

    variances_ : array of float
        Standard deviations of every feature

    """

    def __init__(self,
                 n_rows: int = 10,
                 n_columns: int = 10,
                 init_mode_unsupervised: str = "random",
                 n_iter_unsupervised: int = 1000,
                 train_mode_unsupervised: str = "online",
                 neighborhood_mode_unsupervised: str = "linear",
                 learn_mode_unsupervised: str = "min",
                 distance_metric: str = "euclidean",
                 learning_rate_start=0.5,
                 learning_rate_end=0.05,
                 nbh_dist_weight_mode: str = "pseudo-gaussian",
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
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

    def init_unsuper_som(self):
        """Initialize map."""
        # init node list
        self.node_list_ = np.array(list(
            itertools.product(range(self.n_rows), range(self.n_columns))),
            dtype=int)

        self.max_iterations_ = self.n_iter_unsupervised

        # init radius parameter
        self.radius_max_ = max(self.n_rows, self.n_columns)/2
        self.radius_min_ = 1

        # tqdm paramters
        self.tqdm_params_ = {"disable": not bool(self.verbose),
                             "ncols": 100}

        # init unsupervised SOM in the feature space
        if self.init_mode_unsupervised == "random":
            som = np.random.rand(self.n_rows, self.n_columns, self.X_.shape[1])

        elif self.init_mode_unsupervised == "random_data":
            indices = np.random.randint(
                low=0, high=self.X_.shape[0], size=self.n_rows*self.n_columns)
            som_list = self.X_[indices]
            som = som_list.reshape(
                self.n_rows, self.n_columns, self.X_.shape[1])

        elif self.init_mode_unsupervised == "pca":

            # fixed number of components
            pca = PCA(n_components=2, random_state=self.random_state)

            pca_comp = pca.fit(self.X_).components_

            a_row = np.linspace(-1., 1., self.n_rows)
            a_col = np.linspace(-1., 1., self.n_columns)

            som = np.zeros(
                shape=(self.n_rows, self.n_columns, self.X_.shape[1]))

            for node in self.node_list_:
                som[node[0], node[1], :] = np.add(
                    np.multiply(a_row[node[0]], pca_comp[0]),
                    np.multiply(a_col[node[1]], pca_comp[1]))

        else:
            raise ValueError("Invalid init_mode_unsupervised: "+str(
                self.init_mode_unsupervised))

        self.unsuper_som_ = som

    def fit(self, X, y=None):
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

        self.sample_weights_ = np.full(
            fill_value=1., shape=(len(self.X_), 1))

        self.train_unsupervised_som()
        self.fitted_ = True

        return self

    def train_unsupervised_som(self):
        """Train unsupervised SOM."""
        self.init_unsuper_som()

        if self.train_mode_unsupervised == "online":
            for it in tqdm(range(self.n_iter_unsupervised),
                           desc="unsuper", **self.tqdm_params_):

                # select one input vector & calculate best matching unit (BMU)
                dp = np.random.randint(low=0, high=len(self.X_))
                bmu_pos = self.get_bmu(self.X_[dp], self.unsuper_som_)

                # calculate learning rate and neighborhood function
                learning_rate = self.calc_learning_rate(
                    curr_it=it, mode=self.learn_mode_unsupervised)
                nbh_func = self.calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_unsupervised)

                # calculate distance weight matrix and update weights
                dist_weight_matrix = self.get_nbh_distance_weight_matrix(
                    nbh_func, bmu_pos)
                self.unsuper_som_ = modify_weight_matrix_online(
                    self.unsuper_som_, dist_weight_matrix,
                    true_vector=self.X_[dp],
                    learningrate=learning_rate*self.sample_weights_[dp])

        elif self.train_mode_unsupervised == "batch":
            for it in tqdm(range(self.n_iter_unsupervised),
                           desc="unsuper", **self.tqdm_params_):

                # calculate BMUs
                bmus = self.get_bmus(self.X_)

                # calculate neighborhood function
                nbh_func = self.calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_unsupervised)

                # calculate distance weight matrix for all datapoints
                dist_weight_block = self.get_nbh_distance_weight_block(
                    nbh_func, bmus)

                # update weights
                self.unsuper_som_ = self.modify_weight_matrix_batch(
                    self.unsuper_som_, dist_weight_block, self.X_)

        else:
            raise NotImplementedError("Unsupervised mode not implemented:",
                                      self.train_mode_unsupervised)

        self.set_bmus(self.X_)

    def calc_learning_rate(self, curr_it, mode):
        """Calculate learning rate alpha with 0 <= alpha <= 1.

        Parameters
        ----------
        curr_it : int
            Current iteration count
        mode : str, optional
            Mode of the learning rate (min, exp, expsquare)

        Returns
        -------
        float
            Learning rate

        """
        return decreasing_rate(
            self.learning_rate_start, self.learning_rate_end,
            self.max_iterations_, curr_it, mode)

    def calc_neighborhood_func(self, curr_it, mode):
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
            self.radius_max_, self.radius_min_, self.max_iterations_,
            curr_it, mode)

    def get_bmu(self, datapoint, som_array):
        """Get best matching unit (BMU) for datapoint.

        Parameters
        ----------
        datapoint : np.array, shape=shape[1]
            Datapoint = one row of the dataset X
        som_array : np.array
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])

        Returns
        -------
        tuple, shape = (int, int)
            Position of best matching unit (row, column)

        """
        a = self.get_node_distance_matrix(
            datapoint.astype(np.float64), som_array)

        return np.argwhere(a == np.min(a))[0]

    def get_bmus(self, X, som_array=None):
        """Get Best Matching Units for big datalist.

        Parameters
        ----------
        X : np.array
            List of datapoints
        som_array : np.array, optional (default=`None`)
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
                delayed(self.get_bmu)(dp, som_array)
                for dp in X
            )
        return bmus

    def _partition_bmus(self, X):
        """Private function used to partition bmus between jobs.

        Parameters
        ----------
        X : np.array
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
            n_jobs, n_datapoints // n_jobs, dtype=np.int)

        n_datapoints_per_job[:n_datapoints % n_jobs] += 1
        starts = np.cumsum(n_datapoints_per_job)

        return n_jobs, n_datapoints_per_job.tolist(), [0] + starts.tolist()

    def set_bmus(self, X, som_array=None):
        """Set BMUs in the current SOM object.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples.
        som_array : np.array
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])

        """
        self.bmus_ = self.get_bmus(X=X, som_array=som_array)

    def get_node_distance_matrix(self, datapoint, som_array):
        """Get distance of datapoint and node using Euclidean distance.

        Parameters
        ----------
        datapoint : np.array, shape=(X.shape[1])
            Datapoint = one row of the dataset `X`
        som_array : np.array
            Weight vectors of the SOM,
            shape = (self.n_rows, self.n_columns, X.shape[1])

        Returns
        -------
        distmat : np.array of float
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
                    som_array[node[0], node[1]], datapoint)

        elif self.distance_metric == "mahalanobis":
            for node in self.node_list_:
                som_node = som_array[node[0], node[1]]
                cov = np.cov(np.stack((datapoint, som_node), axis=0),
                             rowvar=False)
                cov_pinv = np.linalg.pinv(cov)   # pseudo-inverse
                distmat[node] = dist.mahalanobis(
                    datapoint, som_node, cov_pinv)

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
                    binarize(datapoint.reshape(1, -1), threshold, copy=True),
                    binarize(som_node.reshape(1, -1), threshold, copy=True))

        return distmat

    def get_nbh_distance_weight_matrix(self, neighborhood_func, bmu_pos):
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
        dist_mat = np.linalg.norm(self.node_list_-bmu_pos, axis=1)

        pseudogaussian = np.exp(-np.divide(np.power(dist_mat, 2),
                                (2 * np.power(neighborhood_func, 2))))

        if self.nbh_dist_weight_mode == "pseudo-gaussian":
            nbh_dist_weight_mat = pseudogaussian.reshape(
                (self.n_rows, self.n_columns, 1))

        elif self.nbh_dist_weight_mode == "mexican-hat":
            mexicanhat = np.multiply(pseudogaussian, np.subtract(1, np.divide(
                np.power(dist_mat, 2), np.power(neighborhood_func, 2))))
            nbh_dist_weight_mat = mexicanhat.reshape(
                (self.n_rows, self.n_columns, 1))

        else:
            raise ValueError("Invalid nbh_dist_weight_mode: "+str(
                self.nbh_dist_weight_mode))

        return nbh_dist_weight_mat

    def get_nbh_distance_weight_block(self, nbh_func, bmus):
        """Calculate distance weight matrix for all datapoints.

        The combination of several distance weight matrices is called
        "block" in the following.

        Parameters
        ----------
        neighborhood_func : float
            Current neighborhood function
        bmu_pos : tuple, shape=(int, int)
            Position of calculated BMU of the current datapoint

        Returns
        -------
        dist_weight_block : np.array of float, shape=(n_rows, n_columns)
            Neighborhood distance weight block between SOM and BMUs

        """
        dist_weight_block = np.zeros(
            (len(bmus), self.n_rows, self.n_columns))

        for i, bmu_pos in enumerate(bmus):
            dist_weight_block[i] = self.get_nbh_distance_weight_matrix(
                nbh_func, bmu_pos).reshape((self.n_rows, self.n_columns))

        return dist_weight_block

    def modify_weight_matrix_batch(self, som_array, dist_weight_matrix, data):
        """Modify weight matrix of the SOM for the online algorithm.

        Parameters
        ----------
        som_array : np.array
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])
        dist_weight_matrix : np.array of float
            Current distance weight of the SOM for the specific node
        data : np.array, optional
            True vector(s)
        learningrate : float
            Current learning rate of the SOM

        Returns
        -------
        np.array
            Weight vector of the SOM after the modification

        """
        # calculate numerator and divisor for the batch formula
        numerator = np.sum(
            [np.multiply(data[i], dist_weight_matrix[i].reshape(
                (self.n_rows, self.n_columns, 1)))
                for i in range(len(data))], axis=0)
        divisor = np.sum(dist_weight_matrix, axis=0).reshape(
            (self.n_rows, self.n_columns, 1))

        # update weights
        old_som = np.copy(som_array)
        new_som = np.divide(
            numerator,
            divisor,
            out=np.full_like(numerator, np.nan),
            where=(divisor != 0))

        # overwrite new nans with old entries
        new_som[np.isnan(new_som)] = old_som[np.isnan(new_som)]
        return new_som

    def transform(self, X, y=None):
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

    def fit_transform(self, X, y=None):
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

    def get_datapoints_from_node(self, node):
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

    def get_clusters(self, X):
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

    def get_u_matrix(self, mode="mean"):
        """Calculate unified distance matrix (u-matrix).

        Parameters
        ----------
        mode : str, optional (default="mean)
            Choice of the averaging algorithm

        Returns
        -------
        np.array
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
            shape=(self.n_rows*2-1, self.n_columns*2-1, 1),
            dtype=float)

        # step 1: fill values between SOM nodes
        self.calc_u_matrix_distances()

        # step 2: fill values at SOM nodes and on diagonals
        self.calc_u_matrix_means()

        return self.u_matrix

    def calc_u_matrix_distances(self):
        """Calculate the Eucl. distances between all neighbored SOM nodes."""
        for u_node in itertools.product(range(self.n_rows*2-1),
                                        range(self.n_columns*2-1)):

            if not (u_node[0] % 2) and (u_node[1] % 2):
                # mean horizontally
                self.u_matrix[u_node] = np.linalg.norm(
                    self.unsuper_som_[u_node[0]//2][u_node[1]//2] -
                    self.unsuper_som_[u_node[0]//2][u_node[1]//2+1])
            elif (u_node[0] % 2) and not (u_node[1] % 2):
                # mean vertically
                self.u_matrix[u_node] = np.linalg.norm(
                    self.unsuper_som_[u_node[0]//2][u_node[1]//2] -
                    self.unsuper_som_[u_node[0]//2+1][u_node[1]//2],
                    axis=0)

    def calc_u_matrix_means(self):
        """Calculate the missing parts of the u-matrix.

        After `calc_u_matrix_distances()`, there are two kinds of entries
        missing: the entries at the positions of the actual SOM nodes and the
        entries in between the distance nodes. Both types of entries are
        calculated in this function.

        """
        for u_node in itertools.product(range(self.n_rows*2-1),
                                        range(self.n_columns*2-1)):

            if not (u_node[0] % 2) and not (u_node[1] % 2):
                # SOM nodes -> mean over 2-4 values

                nodelist = []
                if u_node[0] > 0:
                    nodelist.append((u_node[0]-1, u_node[1]))
                if u_node[0] < self.n_rows*2-2:
                    nodelist.append((u_node[0]+1, u_node[1]))
                if u_node[1] > 0:
                    nodelist.append((u_node[0], u_node[1]-1))
                if u_node[1] < self.n_columns*2-2:
                    nodelist.append((u_node[0], u_node[1]+1))
                self.u_matrix[u_node] = self.get_u_mean(nodelist)

            elif (u_node[0] % 2) and (u_node[1] % 2):
                # mean over four

                self.u_matrix[u_node] = self.get_u_mean([
                    (u_node[0]-1, u_node[1]),
                    (u_node[0]+1, u_node[1]),
                    (u_node[0], u_node[1]-1),
                    (u_node[0], u_node[1]+1)])

    def get_u_mean(self, nodelist):
        """Calculate a mean value of the node entries in `nodelist`.

        Parameters
        ----------
        nodelist : list of tuple (int, int)
            List of nodes on the u-matrix containing distance values

        Returns
        -------
        u_mean : `float`
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
        "tanimoto"}. Note that "tanimoto" tends to be slow.

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

    """

    def __init__(self,
                 n_rows: int = 10,
                 n_columns: int = 10,
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
                 learning_rate_start=0.5,
                 learning_rate_end=0.05,
                 nbh_dist_weight_mode: str = "pseudo-gaussian",
                 missing_label_placeholder=None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        """Initialize SOMEstimator object."""
        super().__init__(n_rows,
                         n_columns,
                         init_mode_unsupervised,
                         n_iter_unsupervised,
                         train_mode_unsupervised,
                         neighborhood_mode_unsupervised,
                         learn_mode_unsupervised,
                         distance_metric,
                         learning_rate_start,
                         learning_rate_end,
                         nbh_dist_weight_mode,
                         n_jobs,
                         random_state,
                         verbose)

        self.init_mode_supervised = init_mode_supervised
        self.n_iter_supervised = n_iter_supervised
        self.train_mode_supervised = train_mode_supervised
        self.neighborhood_mode_supervised = neighborhood_mode_supervised
        self.learn_mode_supervised = learn_mode_supervised
        self.missing_label_placeholder = missing_label_placeholder

    @abstractmethod
    def init_super_som(self):
        """Initialize map."""
        return None

    def fit(self, X, y=None):
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

        return self.fit_estimator(X, y)

    def fit_estimator(self, X, y):
        """Fit supervised SOM to the (checked) input data.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The prediction input samples.
        y : array-like matrix of shape = [n_samples, 1]
            The labels (ground truth) of the input samples

        """
        self.X_ = X
        self.y_ = y

        np.random.seed(seed=self.random_state)

        # supervised case:
        if self.missing_label_placeholder is None:
            self.labeled_indices_ = list(range(len(self.y_)))
            self.sample_weights_ = np.full(
                fill_value=1., shape=(len(self.X_), 1))

        # semi-supervised case:
        else:
            self.labeled_indices_ = np.where(
                self.y_ != self.missing_label_placeholder)[0]
            unlabeled_weight = max(len(self.labeled_indices_) / len(self.y_),
                                   0.1)
            self.sample_weights_ = np.full(
                fill_value=unlabeled_weight, shape=(len(self.X_), 1))
            self.sample_weights_[self.labeled_indices_] = 1.0

        # train SOMs
        self.train_unsupervised_som()
        self.train_supervised_som()

        self.fitted_ = True

        return self

    def predict(self, X, y=None):
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
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, dtype=np.float64)
        y_pred_list = []
        for dp in tqdm(X, desc="predict", **self.tqdm_params_):
            y_pred_list.append(self.calc_estimation_output(dp, mode="bmu"))
        y_pred = np.array(y_pred_list)
        return y_pred

    def calc_estimation_output(self, datapoint, mode="bmu"):
        """Get SOM output for fixed SOM.

        The given datapoint doesn't have to belong to the training set of the
        input SOM.

        Parameters
        ----------
        datapoint : np.array, shape=(X.shape[1])
            Datapoint = one row of the dataset X
        mode : str, optional (default="bmu")
            Mode of the regression output calculation

        Returns
        -------
        estimation_output : object
            Content of SOM node which is linked to the datapoint.
            Classification: the label.
            Regression: the target variable.

        TODO Implement handling of incomplete datapoints
        TODO implement "neighborhood" mode

        """
        bmu_pos = self.get_bmu(datapoint, self.unsuper_som_)
        estimation_output = None

        if mode == "bmu":
            estimation_output = self.super_som_[bmu_pos[0], bmu_pos[1]][0]

        # elif mode == "neighborhood":
        #     nbh_radius = 2.0      # TODO change
        #     nbh_nodes = self.getNeighbors(bmu_pos, nbh_radius)
        #     estimation = np.average(
        #         a=[self.output_som_[node] for node in nbh_nodes],
        #         weights=[self.getNeighborhoodDistanceWeight(nbh_radius,
        #                                                     bmu_pos, node)
        #                  for node in nbh_nodes])

        #     return estimation
        else:
            raise ValueError("Invalid calc_estimation_output mode: "+str(
                mode))

        return estimation_output

    def modify_weight_matrix_supervised(self, dist_weight_matrix,
                                        true_vector=None,
                                        learningrate=None):
        """Modify weights of the supervised SOM, either online or batch.

        Parameters
        ----------
        som_array : np.array
            Weight vectors of the SOM
            shape = (self.n_rows, self.n_columns, X.shape[1])
        dist_weight_matrix : np.array of float
            Current distance weight of the SOM for the specific node
        data : np.array, optional
            True vector(s)
        learningrate : float, optional
            Current learning rate of the SOM

        Returns
        -------
        modify_weight_matrix : np.array
            Weight vector of the SOM after the modification

        """
        modify_weight_matrix = None
        if self.train_mode_supervised == "online":
            modify_weight_matrix = modify_weight_matrix_online(
                self.super_som_, dist_weight_matrix, true_vector=true_vector,
                learningrate=learningrate)

        elif self.train_mode_supervised == "batch":
            modify_weight_matrix = self.modify_weight_matrix_batch(
                som_array=self.super_som_,
                dist_weight_matrix=dist_weight_matrix[self.labeled_indices_],
                data=self.y_[self.labeled_indices_])

        else:
            raise ValueError("Invalid train_mode_supervised: "+str(
                self.train_mode_supervised))

        return modify_weight_matrix

    def train_supervised_som(self):
        """Train supervised SOM."""
        self.set_bmus(self.X_[self.labeled_indices_])
        self.init_super_som()

        if self.train_mode_supervised == "online":
            for it in tqdm(range(self.n_iter_supervised),
                           desc="super", **self.tqdm_params_):

                # select one input vector & calculate best matching unit (BMU)
                dp = self.get_random_datapoint()
                bmu_pos = self.bmus_[dp]

                # calculate learning rate and neighborhood function
                learning_rate = self.calc_learning_rate(
                    curr_it=it, mode=self.learn_mode_supervised)
                nbh_func = self.calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_supervised)

                # calculate distance weight matrix and update weights
                dist_weight_matrix = self.get_nbh_distance_weight_matrix(
                    nbh_func, bmu_pos)
                self.super_som_ = self.modify_weight_matrix_supervised(
                    dist_weight_matrix=dist_weight_matrix,
                    true_vector=self.y_[self.labeled_indices_][dp],
                    learningrate=learning_rate)

        elif self.train_mode_supervised == "batch":
            for it in tqdm(range(self.n_iter_supervised),
                           desc="super", **self.tqdm_params_):

                # calculate BMUs with the unsupervised (!) SOM
                bmus = self.get_bmus(self.X_)

                # calculate neighborhood function
                nbh_func = self.calc_neighborhood_func(
                    curr_it=it, mode=self.neighborhood_mode_supervised)

                # calculate distance weight matrix for all datapoints
                dist_weight_block = self.get_nbh_distance_weight_block(
                    nbh_func, bmus)

                self.super_som_ = self.modify_weight_matrix_supervised(
                    dist_weight_matrix=dist_weight_block)

    def fit_transform(self, X, y=None):
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

    def get_estimation_map(self):
        """Return SOM grid with the estimated value on each node.

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

    def get_random_datapoint(self):
        """Find and return random datapoint from labeled dataset."""
        random_datapoint = None
        if self.missing_label_placeholder is not None:
            random_datapoint = np.random.choice(
                len(self.y_[self.labeled_indices_]))
        else:
            random_datapoint = np.random.randint(low=0, high=len(self.y_))
        return random_datapoint


class SOMRegressor(SOMEstimator, RegressorMixin):
    """Supervised SOM for estimating continuous variables (= regression)."""

    def init_super_som(self):
        """Initialize map for regression."""
        self.max_iterations_ = self.n_iter_supervised

        # check if target variable has dimension 1 or >1
        if len(self.y_.shape) == 1:
            n_regression_vars = 1
        else:
            n_regression_vars = self.y_.shape[1]

        # initialize regression SOM
        if self.init_mode_supervised == "random":
            som = np.random.rand(self.n_rows, self.n_columns,
                                 n_regression_vars)

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
                size=(self.n_rows, self.n_columns, n_regression_vars))

        else:
            raise ValueError("Invalid init_mode_supervised: "+str(
                self.init_mode_supervised))

        self.super_som_ = som


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
        "tanimoto"}. Note that "tanimoto" tends to be slow.

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

    """

    def __init__(self,
                 n_rows: int = 10,
                 n_columns: int = 10,
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
        super().__init__(n_rows,
                         n_columns,
                         init_mode_unsupervised,
                         init_mode_supervised,
                         n_iter_unsupervised,
                         n_iter_supervised,
                         train_mode_unsupervised,
                         train_mode_supervised,
                         neighborhood_mode_unsupervised,
                         neighborhood_mode_supervised,
                         learn_mode_unsupervised,
                         learn_mode_supervised,
                         distance_metric,
                         learning_rate_start,
                         learning_rate_end,
                         nbh_dist_weight_mode,
                         missing_label_placeholder,
                         n_jobs,
                         random_state,
                         verbose)
        self.do_class_weighting = do_class_weighting

    def init_super_som(self):
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
        self.setPlaceholder()

        # check if forbidden class name exists in classes
        if self.placeholder_ in self.classes_:
            raise ValueError("Forbidden class:", self.placeholder_)
        if self.placeholder_ == self.missing_label_placeholder:
            raise ValueError("Forbidden missing_label_placeholder:",
                             self.missing_label_placeholder)

        # class weighting:
        if self.do_class_weighting:
            self.class_weights_ = class_weight.compute_class_weight(
                'balanced', np.unique(self.y_[self.labeled_indices_]),
                self.y_[self.labeled_indices_].flatten())
        else:
            self.class_weights_ = np.ones(shape=self.classes_.shape)

        # initialize classification SOM
        if self.init_mode_supervised == "majority":

            # define dtype
            if self.class_dtype_ in [str, np.str, np.str_]:
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

    def setPlaceholder(self):
        """Set placeholder depending on the class dtype."""
        if self.class_dtype_ in [str, np.str, np.str_]:
            self.placeholder_ = self.placeholder_dict_["str"]
        elif self.class_dtype_ in [int, np.int, np.int_, np.uint8]:
            self.placeholder_ = self.placeholder_dict_["int"]
        elif self.class_dtype_ in [float, np.float, np.float_]:
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

        return self.fit_estimator(X, y)

    def modify_weight_matrix_supervised(self, dist_weight_matrix,
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
            change_class_bool = self.change_class_proba(
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

    def change_class_proba(self, learningrate, dist_weight_matrix,
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
        change_class_proba = learningrate * dist_weight_matrix
        change_class_proba *= class_weight
        random_matrix = np.random.rand(self.n_rows, self.n_columns, 1)
        change_class_bool = random_matrix < change_class_proba
        return change_class_bool
