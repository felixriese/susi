Examples
============

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python3

Regression example
-------------------

In  :bash:`python3`:

.. code:: python3

    import susi

    som = susi.SOMRegressor()
    som.fit(X_train, y_train)
    print(som.score(X_test, y_test))


Classification example
----------------------

In  :bash:`python3`:

.. code:: python3

    import susi

    som = susi.SOMClassifier()
    som.fit(X_train, y_train)
    print(som.score(X_test, y_test))


More examples
-------------

* `examples/SOMClustering <https://github.com/felixriese/susi/blob/main/examples/SOMClustering.ipynb>`_
* `examples/SOMRegressor <https://github.com/felixriese/susi/blob/main/examples/SOMRegressor.ipynb>`_
* `examples/SOMRegressor_semisupervised <https://github.com/felixriese/susi/blob/main/examples/SOMRegressor_semisupervised.ipynb>`_
* `examples/SOMRegressor_multioutput <https://github.com/felixriese/susi/blob/main/examples/SOMRegressor_multioutput.ipynb>`_
* `examples/SOMClassifier <https://github.com/felixriese/susi/blob/main/examples/SOMClassifier.ipynb>`_
* `examples/SOMClassifier_semisupervised <https://github.com/felixriese/susi/blob/main/examples/SOMClassifier_semisupervised.ipynb>`_
