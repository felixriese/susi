"""Test for susi.SOMEstimator.

Usage:
python -m pytest tests/test_SOMEstimator.py

"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import susi  # noqa


@pytest.mark.parametrize(
    "super_som",
    [
        (np.array([[[0], [0.5]], [[1], [2]]])),
    ],
)
def test_get_estimation_map(super_som):
    som = susi.SOMRegressor()  # works the same with SOMClassifier
    som.super_som_ = super_som
    assert np.array_equal(som.get_estimation_map(), super_som)


def test_calc_proba():
    som = susi.SOMRegressor()
    assert som._calc_proba(bmu_pos=(1, 1)) == np.array([1.0])
