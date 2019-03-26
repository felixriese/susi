SUSI: SUpervised Self-organIzing maps in Python
===============================================

Python package for unsupervised and supervised self-organizing maps (SOM)

Description
-----------

We present the SUSI package for Python.
It includes a fully functional SOM for unsupervised and supervised tasks.
The class structure is set up as follows:

- SOMClustering: Unsupervised SOM for clustering

  - SOMEstimator: Base class for supervised SOMs

    - SOMRegressor: Regression SOM
    - SOMClassifier: Classification SOM


:Paper:
    to be published

:License:
    `3-Clause BSD license <LICENSE>`_

:Authors:
    `Felix M. Riese <mailto:github@felixriese.de>`_,
    `Sina Keller <mailto:sina.keller@kit.edu>`_

:Citation: see `Citation`_ and in the `bibtex <bibliography.bib>`_ file

Installation
------------

With PyPi:

.. code:: bash

    pip3 install susi


Manually:

.. code:: bash

    git clone TODO
    cd susi/
    python setup.py install

**Dependencies**

Python 3 with:

* joblib
* numpy
* scikit-learn
* scipy

Usage
-----

Regression in  python3:

.. code:: python3

    import susi

    som = susi.SOMRegressor()
    som.fit(X_train, y_train)
    print(som.score(X_test, y_test))


Classification in  `python3`:

.. code:: python3

    import susi

    som = susi.SOMClassifier()
    som.fit(X_train, y_train)
    print(som.score(X_test, y_test))

Code examples as Jupyter Notebooks:

* `examples/SOMRegressor_Hyperspectral <examples/SOMRegressor_Hyperspectral.ipynb>`_
* `examples/SOMClassifier <examples/SOMClassifier.ipynb>`_
* `examples/SOMClassifier_Salinas <examples/SOMClassifier_Salinas.ipynb>`_

Citation
--------

The bibtex file including both references is available `here <bibliography.bib>`_.

**Paper:**

F. M. Riese and S. Keller, "SUSI: Supervised self-organizing maps for regression and classification in Python", 2019, Submitted to an ISPRS conference.

.. code:: bibtex

    @article{riese2019susi,
        author = {Riese, Felix~M. and Keller, Sina},
        title = {SUSI: Supervised self-organizing maps for regression and classification in Python},
        year = {2019},
        notes = {Submitted to an ISPRS conference},
        TODO arxiv
    }


**Code:**

F. M. Riese, "TODO", [DOI TODO](DOI TODO), 2019.

TODO Badge

.. code:: bibtex

    @misc{riese2019cnn,
      author       = {Riese, Felix~M.},
      title        = {{TODO}},
      year         = {2019},
      publisher    = {Zenodo},
      DOI          = {TODO},
      howpublished = {\href{TODO}{TODO}}
    }
