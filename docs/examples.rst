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

* `examples/SOMClustering <https://github.com/felixriese/susi/blob/master/examples/SOMClustering.ipynb>`_
* `examples/SOMRegressor_Hyperspectral <https://github.com/felixriese/susi/blob/master/examples/SOMRegressor_Hyperspectral.ipynb>`_
* `examples/SOMClassifier <https://github.com/felixriese/susi/blob/master/examples/SOMClassifier.ipynb>`_
* `examples/SOMClassifier_Salinas <https://github.com/felixriese/susi/blob/master/examples/SOMClassifier_Salinas.ipynb>`_
