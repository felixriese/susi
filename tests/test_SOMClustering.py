"""Test for susi.py.

Usage:
python -m pytest tests/test_susi.py

"""
import pytest
import os
import sys
import numpy as np
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import susi


@pytest.mark.parametrize("n_rows,n_columns", [
    (10, 10),
    (12, 15),
])
def test_som_clustering_init(n_rows, n_columns):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_columns)
    assert som_clustering.n_rows == n_rows
    assert som_clustering.n_columns == n_columns
    # assert som_clustering.radius_max == max(n_rows, n_columns) / 2


@pytest.mark.parametrize(
    "learning_rate_start,learning_rate_end,max_it,curr_it,mode,expected", [
        (0.9, 0.1, 800, 34, "min", 0.8197609052582371),
        (0.9, 0.1, 800, 34, "exp", 0.7277042846893071),
    ])
def test_calc_learning_rate(learning_rate_start, learning_rate_end, max_it,
                            curr_it, mode, expected):
    som_clustering = susi.SOMClustering(
        learning_rate_start=learning_rate_start,
        learning_rate_end=learning_rate_end)
    som_clustering.max_iterations_ = max_it
    assert som_clustering.calc_learning_rate(curr_it, mode) == expected


@pytest.mark.parametrize(
    "datapoint,som_array,distance_metric,expected", [
        (np.array([0.3, 2.0, 1.0]),
         np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
                   [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
         "euclidean",
         np.array([[1.4525839, 0.14142136], [2.21585198,  4.64542786]])),
        (np.array([0.3, 2.0, 1.0]),
         np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
                   [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
         "manhattan",
         np.array([[2.9, 2.9], [6.8, 6.8]])),
        (np.array([0.3, 2.0, 1.0]),
         np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
                   [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
         "mahalanobis",
         np.array([[1.41421356, 1.41421356], [1.41421356, 1.41421356]])),
        (np.array([0.3, 2.0, 1.0]),
         np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
                   [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
         "tanimoto",
         np.array([[0.5, 0.5], [0.8, 0.8]])),
    ])
def test_get_node_distance_matrix(datapoint, som_array, distance_metric,
                                  expected):
    som_clustering = susi.SOMClustering()
    som_clustering.distance_metric = distance_metric
    som_clustering.X_ = np.array([datapoint, datapoint])
    som_clustering.n_rows = som_array.shape[0]
    som_clustering.n_columns = som_array.shape[1]
    som_clustering.init_unsuper_som()

    assert np.allclose(som_clustering.get_node_distance_matrix(
        datapoint, som_array), expected, rtol=1e-2)


@pytest.mark.parametrize("radius_max,radius_min,max_it,curr_it,mode,expected", [
    (0.9, 0.1, 800, 34, "min", 0.8197609052582371),
    (0.9, 0.1, 800, 34, "exp", 0.7277042846893071),
])
def test_calc_neighborhood_func(radius_max, radius_min, max_it, curr_it, mode,
                                expected):
    som_clustering = susi.SOMClustering()
    som_clustering.radius_max_ = radius_max
    som_clustering.radius_min_ = radius_min
    som_clustering.max_iterations_ = max_it
    assert som_clustering.calc_neighborhood_func(curr_it, mode) == expected


@pytest.mark.parametrize("a_1,a_2,max_it,curr_it,mode,expected", [
    (0.9, 0.1, 800, 34, "min", 0.8197609052582371),
    (0.9, 0.1, 800, 34, "exp", 0.7277042846893071),
])
def test_decreasing_rate(a_1, a_2, max_it, curr_it, mode, expected):
    assert susi.decreasing_rate(
        a_1, a_2, max_it, curr_it, mode) == expected


@pytest.mark.parametrize("som,expected", [
    (np.array([[0., 1.1, 2.1], [0.3, 2.1, 1.1]]), np.ndarray),
])
def test_init_unsuper_som(som, expected):
    som_clustering = susi.SOMClustering()
    som_clustering.X_ = som
    som_clustering.init_unsuper_som()
    assert type(som_clustering.unsuper_som_) == expected


@pytest.mark.parametrize("som_array,datapoint,expected", [
    (np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([0.3, 2.0, 1.0]), (0, 1)),
])
def test_get_bmu(som_array, datapoint, expected):
    som_clustering = susi.SOMClustering()
    assert np.array_equal(som_clustering.get_bmu(datapoint, som_array), expected)


@pytest.mark.parametrize("X,n_rows,n_columns,random_state,expected", [
    (np.array([[0., 0.1, 0.2], [2.3, 2.1, 2.1]]), 2, 2, 42,
     np.array([[[2.29999999, 2.1, 2.1],
                [1.25232099, 1.18897478, 1.23452604]],
               [[1.25232099, 1.18897478, 1.23452604],
                [2.23083779e-9, 1.00000002e-1, 2.00000002e-1]]])),
])
def test_fit(X, n_rows, n_columns, random_state, expected):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_columns, random_state=random_state)
    som_clustering.fit(X)
    assert type(som_clustering.unsuper_som_) == np.ndarray
    assert som_clustering.unsuper_som_.shape == (n_rows, n_columns, X.shape[1])
    assert np.allclose(som_clustering.unsuper_som_, expected, atol=1e-20)


@pytest.mark.parametrize(
    "n_rows,n_columns,random_state,neighborhood_func,bmu_pos,X,expected", [
        (2, 2, 42, 0.9, (0, 0), np.array([[0., 0.1, 0.2, 0.3], [2.3, 2.1, 2.1, 2.5]]),
         np.array([1., 0.53940751, 0.53940751, 0.29096046])),
])
def test_get_nbh_distance_weight_matrix(n_rows, n_columns, random_state,
                                        neighborhood_func, bmu_pos, X,
                                        expected):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_columns, random_state=random_state)
    som_clustering.X_ = X
    som_clustering.init_unsuper_som()
    assert np.allclose(som_clustering.get_nbh_distance_weight_matrix(
        neighborhood_func, bmu_pos), expected, atol=1e-8)


@pytest.mark.parametrize(
    "n_rows,n_columns,random_state,n_iter_unsupervised, X,learningrate,neighborhood_func,bmu_pos,dp,expected", [
        (2, 2, 42, 2, np.array([[0., 0.1, 0.2], [2.3, 2.1, 2.1]]), 0.7, 0.4, (1,1), 1,
         np.array([[[1.49058628, 1.61686991, 1.52492551],
                    [1.17565903, 0.81530915, 0.81529321]],
                   [[0.81841713, 1.28462067, 1.10945355],
                    [1.91139603, 1.59239575, 1.82413471]]])),
])
def test_modify_weight_matrix(n_rows, n_columns, random_state,
                              n_iter_unsupervised, X, learningrate,
                              neighborhood_func, bmu_pos, dp, expected):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_columns,
        n_iter_unsupervised=n_iter_unsupervised, random_state=random_state)
    som_clustering.fit(X)
    assert np.allclose(som_clustering.modify_weight_matrix(
        som_array=som_clustering.unsuper_som_,
        learningrate=learningrate,
        dist_weight_matrix=som_clustering.get_nbh_distance_weight_matrix(
            neighborhood_func, bmu_pos),
        true_vector=som_clustering.X_[dp]), expected, atol=1e-8)


@pytest.mark.parametrize(
    "n_rows,n_columns,X", [
        (2, 2, np.array([[0., 0.1, 0.2], [2.3, 2.1, 2.1],
         [2.3, 2.1, 2.1], [2.3, 2.1, 2.1]])),
])
def test_transform(n_rows, n_columns, X):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_columns)
    som_clustering.fit(X)
    bmus = som_clustering.transform(X)
    assert(len(bmus) == X.shape[0])
    assert(len(bmus[0]) == 2)


@pytest.mark.parametrize("som_array,X,n_jobs,expected", [
    (np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([[0.3, 2.0, 1.0], [0.3, 2.0, 1.0],
               [0.3, 2.0, 1.0], [1.2, 2.0, 3.4]]),
     1, [(0, 1), (0, 1), (0, 1), (1, 0)]),
    (np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([[0.3, 2.0, 1.0], [0.3, 2.0, 1.0],
               [0.3, 2.0, 1.0], [1.2, 2.0, 3.4]]),
     -1, [(0, 1), (0, 1), (0, 1), (1, 0)]),
])
def test_get_bmus(som_array, X, n_jobs, expected):
    som_clustering = susi.SOMClustering(n_jobs=n_jobs)
    assert np.array_equal(som_clustering.get_bmus(X, som_array), expected)


@pytest.mark.parametrize("som_array,X,n_jobs,expected", [
    (np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([[0.3, 2.0, 1.0], [0.3, 2.0, 1.0],
               [0.3, 2.0, 1.0], [1.2, 2.0, 3.4]]),
     1, [(0, 1), (0, 1), (0, 1), (1, 0)]),
    (np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([[0.3, 2.0, 1.0], [0.3, 2.0, 1.0],
               [0.3, 2.0, 1.0], [1.2, 2.0, 3.4]]),
     -1, [(0, 1), (0, 1), (0, 1), (1, 0)]),
])
def test_set_bmus(som_array, X, n_jobs, expected):
    som_clustering = susi.SOMClustering(n_jobs=n_jobs)
    som_clustering.set_bmus(X, som_array)
    assert np.array_equal(som_clustering.bmus_, expected)


@pytest.mark.parametrize("n_rows,n_columns,som_array,X,node,expected", [
    (3, 3, np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([[0.3, 2.0, 1.0], [0.3, 2.0, 1.0],
               [0.3, 2.0, 1.0], [1.2, 2.0, 3.4]]),
     np.array([0, 0]), []),
    (3, 3, np.array([[[0., 1.1, 2.1], [0.3, 2.1, 1.1]],
               [[1., 2.1, 3.1], [-0.3, -2.1, -1.1]]]),
     np.array([[0.3, 2.0, 1.0], [0.3, 2.0, 1.0],
               [0.3, 2.0, 1.0], [1.2, 2.0, 3.4]]),
     np.array([0, 1]), [0, 1, 2]),
])
def test_get_datapoints_from_node(n_rows, n_columns, som_array, X, node, expected):
    som_clustering = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_columns)
    som_clustering.set_bmus(X, som_array)
    assert(np.array_equal(som_clustering.get_datapoints_from_node(node), expected))
