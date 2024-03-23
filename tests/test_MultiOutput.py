"""Test for MultiOutput.

Usage:
python -m pytest tests/test_MultiOutput.py

"""

import os
import sys

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import susi  # noqa

# define test dataset
cali = fetch_california_housing()
X = cali.data
y = np.array([cali.target, cali.target]).T


def test_MultiOutputRegressor():
    mor = MultiOutputRegressor(estimator=susi.SOMRegressor(n_jobs=2), n_jobs=2)
    mor.fit(X, y)
