"""Test for susi.SOMClustering.

Usage:
python -m pytest tests/test_SOMClustering.py

"""

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_biclusters

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import susi  # noqa

test_data, _, _ = make_biclusters((100, 10), 3)


@pytest.mark.parametrize(
    "n_rows,n_columns",
    [
        (10, 10),
        (12, 15),
    ],
)
def test_som_clustering_init(n_rows, n_columns):
    som_clustering = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns)
    assert som_clustering.n_rows == n_rows
    assert som_clustering.n_columns == n_columns


@pytest.mark.parametrize(
    "learning_rate_start,learning_rate_end,max_it,curr_it,mode,expected",
    [
        (0.9, 0.1, 800, 34, "min", 0.8197609052582371),
        (0.9, 0.1, 800, 34, "exp", 0.7277042846893071),
    ],
)
def test_calc_learning_rate(
    learning_rate_start, learning_rate_end, max_it, curr_it, mode, expected
):
    som_clustering = susi.SOMClustering(
        learning_rate_start=learning_rate_start,
        learning_rate_end=learning_rate_end,
    )
    som_clustering.max_iterations_ = max_it
    assert np.allclose(
        som_clustering._calc_learning_rate(curr_it, mode), expected
    )


@pytest.mark.parametrize(
    "datapoint,som_array,distance_metric,expected",
    [
        (
            np.array([0.3, 2.0, 1.0]),
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            "euclidean",
            np.array([[1.4525839, 0.14142136], [2.21585198, 4.64542786]]),
        ),
        (
            np.array([0.3, 2.0, 1.0]),
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            "manhattan",
            np.array([[2.9, 2.9], [6.8, 6.8]]),
        ),
        (
            np.array([0.3, 2.0, 1.0]),
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            "mahalanobis",
            np.array([[1.41421356, 1.41421356], [1.41421356, 1.41421356]]),
        ),
        (
            np.array([0.3, 2.0, 1.0]),
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            "tanimoto",
            np.array([[0.5, 0.5], [0.8, 0.8]]),
        ),
        (
            np.array([0.3, 2.0, 1.0]),
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            "spectralangle",
            np.array([[0.93360572, 0.93360572], [2.00637762, 2.00637762]]),
        ),
    ],
)
def test_get_node_distance_matrix(
    datapoint, som_array, distance_metric, expected
):
    som_clustering = susi.SOMClustering()
    som_clustering.distance_metric = distance_metric
    som_clustering.X_ = np.array([datapoint, datapoint])
    som_clustering.n_rows = som_array.shape[0]
    som_clustering.n_columns = som_array.shape[1]
    som_clustering._init_unsuper_som()
    assert np.allclose(
        som_clustering._get_node_distance_matrix(datapoint, som_array),
        expected,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "radius_max,radius_min,max_it,curr_it,mode,expected",
    [
        (0.9, 0.1, 800, 34, "min", 0.8197609052582371),
        (0.9, 0.1, 800, 34, "exp", 0.7277042846893071),
    ],
)
def test_calc_neighborhood_func(
    radius_max, radius_min, max_it, curr_it, mode, expected
):
    som_clustering = susi.SOMClustering()
    som_clustering.radius_max_ = radius_max
    som_clustering.radius_min_ = radius_min
    som_clustering.max_iterations_ = max_it
    assert np.allclose(
        som_clustering._calc_neighborhood_func(curr_it, mode), expected
    )


@pytest.mark.parametrize(
    "a_1,a_2,max_it,curr_it,mode,expected",
    [
        (0.9, 0.1, 800, 34, "min", 0.8197609052582371),
        (0.9, 0.1, 800, 34, "exp", 0.7277042846893071),
        (0.9, 0.1, 800, 34, "expsquare", 0.8919084683204536),
        (0.9, 0.1, 800, 34, "linear", 0.866),
        (0.9, 0.1, 800, 34, "inverse", 0.026470588235294117),
        (0.9, 0.1, 800, 34, "root", 0.9955321885817805),
        (0.9, 0.1, 800, 34, "testerror", 0.7277042846893071),
    ],
)
def test_decreasing_rate(a_1, a_2, max_it, curr_it, mode, expected):
    if mode == "testerror":
        with pytest.raises(ValueError):
            assert (
                susi.decreasing_rate(
                    a_1,
                    a_2,
                    iteration_max=max_it,
                    iteration=curr_it,
                    mode=mode,
                )
                == expected
            )

    else:
        rate = susi.decreasing_rate(
            a_1, a_2, iteration_max=max_it, iteration=curr_it, mode=mode
        )
        assert np.allclose(rate, expected, rtol=1e-2)


@pytest.mark.parametrize(
    "X,init_mode",
    [
        (np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]), "random"),
        (np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]), "random_data"),
        (np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]), "pca"),
        (np.array([[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]]), "rrandom"),
    ],
)
def test_init_unsuper_som(X, init_mode):
    som_clustering = susi.SOMClustering(init_mode_unsupervised=init_mode)
    som_clustering.X_ = X

    if init_mode in ["random", "random_data", "pca"]:
        som_clustering._init_unsuper_som()

        # test type
        assert isinstance(som_clustering.unsuper_som_, np.ndarray)

        # test shape
        n_rows = som_clustering.n_rows
        n_columns = som_clustering.n_columns
        assert som_clustering.unsuper_som_.shape == (
            n_rows,
            n_columns,
            X.shape[1],
        )

    else:
        with pytest.raises(ValueError):
            som_clustering._init_unsuper_som()


@pytest.mark.parametrize(
    "som_array,datapoint,expected",
    [
        (
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array([0.3, 2.0, 1.0]),
            (0, 1),
        ),
    ],
)
def test_get_bmu(som_array, datapoint, expected):
    som_clustering = susi.SOMClustering()
    assert np.array_equal(
        som_clustering.get_bmu(datapoint, som_array), expected
    )


@pytest.mark.parametrize(
    "X,n_rows,n_columns,train_mode_unsupervised,random_state,expected",
    [
        (
            np.array([[0.0, 0.1, 0.2], [2.3, 2.1, 2.1]]),
            2,
            2,
            "online",
            42,
            np.array(
                [
                    [
                        [1.72651971, 1.60132149, 1.62625542],
                        [1.2091674, 1.15144991, 1.19887742],
                    ],
                    [
                        [1.2091674, 1.15144991, 1.19887742],
                        [0.66132515, 0.67506535, 0.74631208],
                    ],
                ]
            ),
        ),
        (
            np.array([[0.0, 0.1, 0.2], [2.3, 2.1, 2.1]]),
            2,
            2,
            "batch",
            42,
            np.array(
                [
                    [[1.68143473, 1.56211716, 1.5890113], [1.15, 1.1, 1.15]],
                    [[1.15, 1.1, 1.15], [0.61856527, 0.63788284, 0.7109887]],
                ]
            ),
        ),
    ],
)
def test_fit(
    X, n_rows, n_columns, train_mode_unsupervised, random_state, expected
):
    som = susi.SOMClustering(
        n_rows=n_rows,
        n_columns=n_columns,
        train_mode_unsupervised=train_mode_unsupervised,
        random_state=random_state,
    )

    som.fit(X)
    assert isinstance(som.unsuper_som_, np.ndarray)
    assert som.unsuper_som_.shape == (n_rows, n_columns, X.shape[1])
    assert np.allclose(som.unsuper_som_, expected, atol=1e-20)

    with pytest.raises(NotImplementedError):
        som = susi.SOMClustering(train_mode_unsupervised="alsdkf")
        som.fit(X)


@pytest.mark.parametrize(
    (
        "n_rows,n_columns,random_state,neighborhood_func,bmu_pos,X,"
        "mode,expected"
    ),
    [
        (
            2,
            2,
            42,
            0.9,
            (0, 0),
            np.array([[0.0, 0.1, 0.2, 0.3], [2.3, 2.1, 2.1, 2.5]]),
            "pseudo-gaussian",
            np.array([[[1.0], [0.53940751]], [[0.53940751], [0.29096046]]]),
        ),
        (
            2,
            2,
            42,
            0.9,
            (0, 0),
            np.array([[0.0, 0.1, 0.2, 0.3], [2.3, 2.1, 2.1, 2.5]]),
            "mexican-hat",
            np.array([[[1.0], [-0.12652769]], [[-0.12652769], [-0.42746043]]]),
        ),
    ],
)
def test_get_nbh_distance_weight_matrix(
    n_rows,
    n_columns,
    random_state,
    neighborhood_func,
    bmu_pos,
    X,
    mode,
    expected,
):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows,
        n_columns=n_columns,
        nbh_dist_weight_mode=mode,
        random_state=random_state,
    )
    som_clustering.X_ = X
    som_clustering._init_unsuper_som()
    print(
        som_clustering._get_nbh_distance_weight_matrix(
            neighborhood_func, bmu_pos
        )
    )
    print(expected)
    assert np.allclose(
        som_clustering._get_nbh_distance_weight_matrix(
            neighborhood_func, bmu_pos
        ),
        expected,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    (
        "n_rows,n_columns,random_state,n_iter_unsupervised, X,learning_rate,"
        "neighborhood_func,bmu_pos,dp,expected"
    ),
    [
        (
            2,
            2,
            42,
            2,
            np.array([[0.0, 0.1, 0.2], [2.3, 2.1, 2.1]]),
            0.7,
            0.4,
            (1, 1),
            1,
            np.array(
                [
                    [
                        [1.49058628, 1.61686991, 1.52492551],
                        [1.26125694, 0.91311475, 0.91310002],
                    ],
                    [
                        [0.93121244, 1.34669682, 1.18486546],
                        [1.9329369, 1.62053297, 1.83942631],
                    ],
                ]
            ),
        ),
    ],
)
def test_modify_weight_matrix_online(
    n_rows,
    n_columns,
    random_state,
    n_iter_unsupervised,
    X,
    learning_rate,
    neighborhood_func,
    bmu_pos,
    dp,
    expected,
):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows,
        n_columns=n_columns,
        n_iter_unsupervised=n_iter_unsupervised,
        random_state=random_state,
    )
    som_clustering.fit(X)
    assert np.allclose(
        susi.modify_weight_matrix_online(
            som_array=som_clustering.unsuper_som_,
            learning_rate=learning_rate,
            dist_weight_matrix=som_clustering._get_nbh_distance_weight_matrix(
                neighborhood_func, bmu_pos
            ),
            true_vector=som_clustering.X_[dp],
        ),
        expected,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("X,nbh_func,bmus,expected"),
    [
        (
            np.array([[0.0, 0.1, 0.2], [2.3, 2.1, 2.1]]),
            0.4,
            np.array([[1, 1], [1, 0]]),
            np.array(
                [
                    [
                        [2.20319823, 2.01582454, 2.02003332],
                        [0.09680177, 0.18417546, 0.27996668],
                    ],
                    [
                        [2.20319823, 2.01582454, 2.02003332],
                        [0.09680177, 0.18417546, 0.27996668],
                    ],
                ]
            ),
        ),
    ],
)
def test_modify_weight_matrix_batch(X, nbh_func, bmus, expected):
    som = susi.SOMClustering(
        n_rows=2, n_columns=2, n_iter_unsupervised=5, random_state=42
    )
    som.fit(X)

    # calculate distance weight matrix for all datapoints
    dist_weight_block = np.zeros((len(X), som.n_rows, som.n_columns))
    for i, bmu_pos in enumerate(bmus):
        dist_weight_block[i] = som._get_nbh_distance_weight_matrix(
            nbh_func, bmu_pos
        ).reshape((som.n_rows, som.n_columns))

    new_som = som._modify_weight_matrix_batch(
        som_array=som.unsuper_som_,
        dist_weight_matrix=dist_weight_block,
        data=som.X_,
    )
    assert np.allclose(new_som, expected, atol=1e-8)


@pytest.mark.parametrize(
    "n_rows,n_columns,X",
    [
        (
            2,
            2,
            np.array(
                [
                    [0.0, 0.1, 0.2],
                    [2.3, 2.1, 2.1],
                    [2.3, 2.1, 2.1],
                    [2.3, 2.1, 2.1],
                ]
            ),
        ),
    ],
)
def test_transform(n_rows, n_columns, X):
    som_clustering = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns)
    som_clustering.fit(X)
    bmus = som_clustering.transform(X)
    assert len(bmus) == X.shape[0]
    assert len(bmus[0]) == 2


@pytest.mark.parametrize(
    "n_rows,n_columns,X",
    [
        (
            2,
            2,
            np.array(
                [
                    [0.0, 0.1, 0.2],
                    [2.3, 2.1, 2.1],
                    [2.3, 2.1, 2.1],
                    [2.3, 2.1, 2.1],
                ]
            ),
        ),
    ],
)
def test_fit_transform(n_rows, n_columns, X):
    som_clustering = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns)
    bmus = som_clustering.fit_transform(X)
    assert len(bmus) == X.shape[0]
    assert len(bmus[0]) == 2


@pytest.mark.parametrize(
    "som_array,X,n_jobs,expected",
    [
        (
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array(
                [
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [1.2, 2.0, 3.4],
                ]
            ),
            1,
            [(0, 1), (0, 1), (0, 1), (1, 0)],
        ),
        (
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array(
                [
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [1.2, 2.0, 3.4],
                ]
            ),
            -1,
            [(0, 1), (0, 1), (0, 1), (1, 0)],
        ),
    ],
)
def test_get_bmus(som_array, X, n_jobs, expected):
    som_clustering = susi.SOMClustering(n_jobs=n_jobs)
    assert np.array_equal(som_clustering.get_bmus(X, som_array), expected)


@pytest.mark.parametrize(
    "som_array,X,n_jobs,expected",
    [
        (
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array(
                [
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [1.2, 2.0, 3.4],
                ]
            ),
            1,
            [(0, 1), (0, 1), (0, 1), (1, 0)],
        ),
        (
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array(
                [
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [1.2, 2.0, 3.4],
                ]
            ),
            -1,
            [(0, 1), (0, 1), (0, 1), (1, 0)],
        ),
    ],
)
def test_set_bmus(som_array, X, n_jobs, expected):
    som_clustering = susi.SOMClustering(n_jobs=n_jobs)
    som_clustering._set_bmus(X, som_array)
    assert np.array_equal(som_clustering.bmus_, expected)


@pytest.mark.parametrize(
    "n_rows,n_columns,som_array,X,node,expected",
    [
        (
            3,
            3,
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array(
                [
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [1.2, 2.0, 3.4],
                ]
            ),
            np.array([0, 0]),
            [],
        ),
        (
            3,
            3,
            np.array(
                [
                    [[0.0, 1.1, 2.1], [0.3, 2.1, 1.1]],
                    [[1.0, 2.1, 3.1], [-0.3, -2.1, -1.1]],
                ]
            ),
            np.array(
                [
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [0.3, 2.0, 1.0],
                    [1.2, 2.0, 3.4],
                ]
            ),
            np.array([0, 1]),
            [0, 1, 2],
        ),
    ],
)
def test_get_datapoints_from_node(
    n_rows, n_columns, som_array, X, node, expected
):
    som = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns)
    som._set_bmus(X, som_array)
    assert np.array_equal(som.get_datapoints_from_node(node), expected)


@pytest.mark.parametrize(
    "n_rows,n_columns,mode",
    [
        (3, 3, "mean"),
        (10, 5, "median"),
        (100, 3, "min"),
        (30, 30, "max"),
    ],
)
def test_get_u_matrix(n_rows, n_columns, mode):
    som = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns)
    som.fit(test_data)
    u_matrix = som.get_u_matrix(mode=mode)
    assert isinstance(u_matrix, np.ndarray)
    assert u_matrix.shape == (n_rows * 2 - 1, n_columns * 2 - 1, 1)


def test_get_clusters():
    som = susi.SOMClustering()
    som.fit(test_data)
    clusters = som.get_clusters(test_data)
    assert len(clusters) == len(test_data)
    assert len(clusters[0]) == 2


class TestGetQuantizationError:
    def test_with_default_data(self) -> None:
        # given
        som = susi.SOMClustering()
        som.fit(test_data)

        # when
        qerror = som.get_quantization_error()

        # then
        assert qerror < 0.05

    def test_with_explicit_data(self, mocker) -> None:
        # given
        som = susi.SOMClustering()
        X = np.array(
            [
                [8, 9, 7, 8, 6, 3, 0, 4, 1, 6],
                [3, 2, 1, 2, 5, 7, 0, 0, 3, 3],
                [0, 7, 5, 1, 4, 5, 7, 5, 8, 5],
                [9, 1, 1, 7, 5, 9, 8, 9, 3, 3],
                [2, 7, 7, 2, 3, 3, 3, 3, 3, 3],
            ]
        )
        som.fit(X)
        weights_per_datapoint = [
            [8, 5, 0, 4, 5, 7, 0, 2, 0, 2],
            [4, 0, 7, 4, 8, 0, 4, 2, 2, 8],
            [5, 3, 1, 2, 7, 9, 9, 8, 8, 7],
            [8, 6, 8, 7, 7, 5, 4, 7, 1, 1],
            [6, 2, 3, 5, 7, 5, 9, 5, 9, 0],
        ]
        mocker.patch.object(
            som,
            "_get_weights_per_datapoint",
            return_value=weights_per_datapoint,
        )

        # when
        qerror = som.get_quantization_error(X)

        # then
        assert qerror == 11.45650021348017
