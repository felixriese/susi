"""Test for MultiOutput.

Usage:
python -m pytest tests/test_MultiOutput.py

"""
import os
import sys

import numpy as np
from sklearn.datasets import load_boston
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import susi


# define test dataset
boston = load_boston()
X = boston.data
y = np.array([boston.target, boston.target]).T


def test_MultiOutputRegressor():
    mor = MultiOutputRegressor(
        estimator=susi.SOMRegressor(n_jobs=2),
        n_jobs=2
    )
    mor.fit(X, y)
