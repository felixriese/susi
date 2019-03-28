.. image:: https://badge.fury.io/py/susi.svg
    :target: https://pypi.org/project/susi/
    :alt: PyPi - Code Version

.. image:: https://img.shields.io/pypi/pyversions/susi.svg
    :target: https://pypi.org/project/susi/
    :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/l/susi.svg
    :target: https://github.com/felixriese/susi/blob/master/LICENSE
    :alt: PyPI - License

.. image:: https://travis-ci.org/felixriese/susi.svg?branch=master
    :target: https://travis-ci.org/felixriese/susi
    :alt: Travis.CI Status

.. image:: https://readthedocs.org/projects/susi/badge/?version=latest
    :target: https://susi.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/felixriese/susi/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/felixriese/susi
    :alt: Codecov

.. image:: https://api.codacy.com/project/badge/Grade/d304689a7364437db1ef998cf7765f5a
	:target: https://app.codacy.com/app/felixriese/susi
	:alt: Codacy Badge

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

:License:
    `3-Clause BSD license <LICENSE>`_

:Author:
    `Felix M. Riese <mailto:github@felixriese.de>`_

:Citation:
    see `Citation`_ and in the `bibtex <bibliography.bib>`_ file

:Documentation:
    `read the docs <https://susi.readthedocs.io/en/latest/readme.html>`_

:Paper:
    `arXiv:1903.11114 <https://arxiv.org/abs/1903.11114>`_


Installation
------------

With PyPi:

.. code:: bash

    pip3 install susi


Manually:

.. code:: bash

    git clone https://github.com/felixriese/susi.git
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

Felix M. Riese and S. Keller, "SUSI: Supervised Self-Organizing Maps for Regression and Classification in Python", `arXiv:1903.11114 <https://arxiv.org/abs/1903.11114>`_, 2019. Submitted to an ISPRS conference.

.. code:: bibtex

    @article{riesekeller2019susi,
        author = {Riese, Felix~M. and Keller, Sina},
        title = {SUSI: Supervised Self-Organizing Maps for Regression and Classification in Python},
        year = {2019},
        notes = {Submitted to an ISPRS conference},
	    archivePrefix = {arXiv},
   	    eprint = {1903.11114},
   	    primaryClass = {cs.LG},
   	    url = {https://arxiv.org/abs/1903.11114}
    }

**Code:**

Felix M. Riese, "SUSI: SUpervised Self-organIzing maps in Python", `10.5281/zenodo.2609130 <https://doi.org/10.5281/zenodo.2609130>`_, 2019.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2609130.svg
   :target: https://doi.org/10.5281/zenodo.2609130

.. code:: bibtex

    @misc{riese2019susicode,
        author = {Riese, Felix~M.},
        title = {{SUSI: SUpervised Self-organIzing maps in Python}},
        year = {2019},
        DOI = {10.5281/zenodo.2609130},
        publisher = {Zenodo},
        howpublished = {\href{https://doi.org/10.5281/zenodo.2609130}{doi.org/10.5281/zenodo.2609130}}
    }
