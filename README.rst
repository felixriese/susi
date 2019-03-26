.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2609131.svg
    :target: https://doi.org/10.5281/zenodo.2609131
    :alt: Zenodo DOI

.. image:: https://travis-ci.org/felixriese/susi.svg?branch=master
    :target: https://travis-ci.org/felixriese/susi
    :alt: Travis.CI

.. image:: https://readthedocs.org/projects/susi/badge/?version=latest
    :target: https://susi.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

|

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
    Felix M. Riese and S. Keller, "SUSI: Supervised Self-Organizing Maps for Regression and Classification in Python", 2019, Submitted to an ISPRS conference.

:License:
    `3-Clause BSD license <LICENSE>`_

:Authors:
    `Felix M. Riese <mailto:github@felixriese.de>`_,
    `Sina Keller <mailto:sina.keller@kit.edu>`_

:Citation:
    see `Citation`_ and in the `bibtex <bibliography.bib>`_ file

:Documentation:
    `read the docs <https://susi.readthedocs.io/en/latest/readme.html>`_

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

Felix M. Riese and S. Keller, "SUSI: Supervised Self-Organizing Maps for Regression and Classification in Python", 2019, Submitted to an ISPRS conference.

.. code:: bibtex

    @article{riesekeller2019susi,
        author = {Riese, Felix~M. and Keller, Sina},
        title = {SUSI: Supervised Self-Organizing Maps for Regression and Classification in Python},
        year = {2019},
        notes = {Submitted to an ISPRS conference},
    }


**Code:**

Felix M. Riese, "SUSI: SUpervised Self-organIzing maps in Python", [10.5281/zenodo.2609130](https://doi.org/10.5281/zenodo.2609130), 2019.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2609131.svg
   :target: https://doi.org/10.5281/zenodo.2609131

.. code:: bibtex

    @misc{riese2019susicode,
        author = {Riese, Felix~M.},
        title = {{SUSI: SUpervised Self-organIzing maps in Python}},
        year = {2019},
        DOI = {10.5281/zenodo.2609130},
        publisher = {Zenodo},
        howpublished = {\href{https://doi.org/10.5281/zenodo.2609130}{doi.org/10.5281/zenodo.2609130}}
    }
