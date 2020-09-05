Hyperparameters
================

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python3


In the following, the most important hyperparameters of the SuSi package are
described. The default hyperparameter settings are a good start, but can
always be optimized. You can do that yourself or through an optimization.
The commonly used hyperparameter settings are taken from [RieseEtAl2020]_.


Grid Size (n_rows, n_columns)
-------------------------------

The grid size of a SOM is defined with the parameters :bash:`n_rows` and
:bash:`n_columns`, the numbers of rows and columns.
The choice of the grid size depends on several trade-offs.

**Characteristics of a larger grid:**

- Better adaption on complex problems (good!)
- Better/smoother visualization capabilities (good!)
- More possible overtraining (possibly bad)
- Larger training time (bad if very limited resources)

**Our Recommendation:**
    We suggest to start with a small grid, meaning :bash:`5 x 5`, and extending
    this grid size while tracking the test and training error metrics.
    We consider SOM grids as "large" with a grid size of about :bash:`100 x 100`
    and more. Non-square SOM grids can also be helpful for specific problems.
    Commonly used grid sizes are :bash:`50 x 50` to :bash:`80 x 80`.


Number of iterations and training mode
----------------------------------------

The number of iterations (:bash:`n_iter_unsupervised` and
:bash:`n_iter_supervised`) depends on the training mode (
:bash:`train_mode_unsupervised` and :bash:`train_mode_supervised`).

**Our Recommendation (Online Mode)**
    Use the *online* mode. If your dataset is small (< 1000 datapoints), use
    10 000 iterations for the unsupervised SOM and 5000 iterations for the
    supervised SOM as start values. If your dataset is significantly larger,
    use significantly more iterations. Commonly used value ranges are  for the
    unsupervised SOM 10 000 to 60 000 and for the (semi-)supervised SOM about
    20 000 to 70 000 in the *online* mode.

**Our Recommendation (Batch Mode)**
    To be evaluated.

Neighborhood Distance Weight, Neighborhood Function, and Learning Rate
------------------------------------------------------------------------

The hyperparameters around the neighborhood mode
(:bash:`neighborhood_mode_unsupervised` + :bash:`neighborhood_mode_supervised`)
and the learning rate (:bash:`learn_mode_unsupervised`,
:bash:`learn_mode_supervised`, :bash:`learning_rate_start`, and
:bash:`learning_rate_end`) depend on the neighborhood distance weight formula
:bash:`nbh_dist_weight_mode`. Two different modes are implemented so far:
:bash:`pseudo-gaussian` and :bash:`mexican-hat`.

**Our Recommendation (Pseudo-Gaussian):**
    Use the :bash:`pseudo-gaussian` neighborhood distance weight with the
    default formulas for the neighborhood mode and the learning rate. The most
    influence, from our experiences, comes from the start (and end) value of
    the learning rate (:bash:`learning_rate_start`, and
    :bash:`learning_rate_end`). They should be optimized. Commonly used
    formula are :bash:`linear` and :bash:`min` for the neighborhood mode,
    :bash:`min` and :bash:`exp` for the learning rate mode, start values from
    0.3 to 0.8 and end values from 0.1 to 0.005.


**Our Recommendation (Mexican-Hat):**
    To be evaluated.

Distance Metric
-----------------

In the following, we give recommendations for the different distance metrics.
Implemented into the SuSi package are the following metrics:

* Euclidean Distance, see `Wikipedia "Euclidean Distance" <https://en.wikipedia.org/wiki/Euclidean_distance>`_
* Manhattan Distance, see `Wikipedia "Taxicab geometry" <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
* Mahalanobis Distance, see `Wikipedia "Mahalanobis distance" <https://en.wikipedia.org/wiki/Mahalanobis_distance>`_
* Tanimoto Distance, see `Wikipedia "Jaccard index - Tanimoto similarity and distance <https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance>`_
* Spectral Angle Distance, see e.g. [YuhasEtAl1992]_

**Our Recommendation:**
    Depending on the bandwidth, number of channels, and overlap of the spectral
    channels, a distance metric can have a significant impact on the training.
    While we have solely relied on the Euclidean distance in [RieseEtAl2020]_,
    we have seen in other, not SOM-related articles, that the Mahalanobis and
    Spectral Angle distance were helpful in the spectral separation of classes.

Hyperparameter optimization
---------------------------

Possible ways to find optimal hyperparameters for a problem are a grid search
or randomized search. Because the SuSi package is developed according to
several scikit-learn guidelines, it can be used with:

- `scikit-learn.model_selection.GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
- `scikit-learn.model_selection.RandomizedSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_

For example, the randomized search can be applied as follows in :bash:`Python3`:

.. code:: python3

    import susi
    from sklearn.datasets import load_iris
    from sklearn.model_selection import RandomizedSearchCV

    iris = load_iris()
    param_grid = {
        "n_rows": [5, 10, 20],
        "n_columns": [5, 20, 40],
        "learning_rate_start": [0.5, 0.7, 0.9],
        "learning_rate_end": [0.1, 0.05, 0.005],
    }
    som = susi.SOMRegressor()
    clf = RandomizedSearchCV(som, param_grid, random_state=1)
    clf.fit(iris.data, iris.target)
    print(clf.best_params_)



References
------------

.. [RieseEtAl2020] F. M. Riese, S. Keller and S. Hinz, "Supervised and
    Semi-Supervised Self-Organizing Maps for Regression and Classification
    Focusing on Hyperspectral Data", *Remote Sensing*, vol. 12, no. 1, 2020.
    `MDPI Link <https://www.mdpi.com/2072-4292/12/1/7>`_
.. [YuhasEtAl1992] R. H. Yuhas, A. F. Goetz, & J. W. Boardman (1992).
    Discrimination among semi-arid landscape endmembers using the spectral
    angle mapper (SAM) algorithm.
    `NASA Link <https://ntrs.nasa.gov/citations/19940012238>`_
