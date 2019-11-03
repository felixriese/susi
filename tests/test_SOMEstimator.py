"""Test for susi.SOMEstimator.

Usage:
python -m pytest tests/test_SOMEstimator.py

"""
import pytest
import os
import sys
import numpy as np
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import susi


@pytest.mark.parametrize(
    "super_som", [
        (np.array([[[0], [0.5]], [[1], [2]]])),
    ])
def test_get_estimation_map(super_som):
    som = susi.SOMRegressor()  # works the same with SOMClassifier
    som.super_som_ = super_som
    assert np.array_equal(som.get_estimation_map(), super_som)
