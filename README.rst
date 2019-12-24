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

SuSi: SUpervised Self-organIzing maps in Python
===============================================

Python package for unsupervised, supervised and semi-supervised self-organizing maps (SOM)

Description
-----------

We present the SuSi package for Python.
It includes a fully functional SOM for unsupervised, supervised and semi-supervised tasks.
The class structure is set up as follows:

- SOMClustering: Unsupervised SOM for clustering

  - SOMEstimator: Base class for supervised and semi-supervised SOMs

    - SOMRegressor: Regression SOM
    - SOMClassifier: Classification SOM

:License:
    `3-Clause BSD license <LICENSE>`_

:Author:
    `Felix M. Riese <mailto:github@felixriese.de>`_

:Citation:
    see `Citation`_ and in the `bibtex <https://github.com/felixriese/susi/blob/master/bibliography.bib>`_ file

:Documentation:
    `Documentation <https://susi.readthedocs.io/en/latest/index.html>`_

:Installation:
    `Installation guidelines <https://susi.readthedocs.io/en/latest/install.html>`_

:Paper:
    `F. M. Riese, S. Keller and S. Hinz in Remote Sensing <https://www.mdpi.com/2072-4292/12/1/7>`_


Installation
------------

.. code:: bash

    pip install susi

More information can be found in the `installation guidelines <https://susi.readthedocs.io/en/latest/install.html>`_.

Examples
--------

A collection of code examples can be found in `the documentation <https://susi.readthedocs.io/en/latest/examples.html>`_.
Code examples as Jupyter Notebooks can be found here:

* `examples/SOMClustering <https://github.com/felixriese/susi/blob/master/examples/SOMClustering.ipynb>`_
* `examples/SOMRegressor <https://github.com/felixriese/susi/blob/master/examples/SOMRegressor.ipynb>`_
* `examples/SOMRegressor_semisupervised <https://github.com/felixriese/susi/blob/master/examples/SOMRegressor_semisupervised.ipynb>`_
* `examples/SOMClassifier <https://github.com/felixriese/susi/blob/master/examples/SOMClassifier.ipynb>`_
* `examples/SOMClassifier_semisupervised <https://github.com/felixriese/susi/blob/master/examples/SOMClassifier_semisupervised.ipynb>`_

Citation
--------

The bibtex file including both references is available `here <https://github.com/felixriese/susi/blob/master/bibliography.bib>`_.

**Paper:**

F. M. Riese, S. Keller and S. Hinz, "Supervised and Semi-Supervised Self-Organizing
Maps for Regression and Classification Focusing on Hyperspectral Data",
*Remote Sensing*, vol. 12, no. 1, 2019.

.. code:: bibtex

    @article{riese2019supervised,
        author = {Riese, Felix~M. and Keller, Sina and Hinz, Stefan},
        title = {{Supervised and Semi-Supervised Self-Organizing Maps for
                  Regression and Classification Focusing on Hyperspectral Data}},
        journal = {Remote Sensing},
        year = {2019},
        volume = {12},
        number = {1},
        article-number = {7},
        URL = {https://www.mdpi.com/2072-4292/12/1/7},
        ISSN = {2072-4292},
        DOI = {10.3390/rs12010007}
    }

**Code:**

Felix M. Riese, "SuSi: SUpervised Self-organIzing maps in Python",
`10.5281/zenodo.2609130 <https://doi.org/10.5281/zenodo.2609130>`_, 2019.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2609130.svg
   :target: https://doi.org/10.5281/zenodo.2609130

.. code:: bibtex

    @misc{riese2019susicode,
        author = {Riese, Felix~M.},
        title = {{SuSi: SUpervised Self-organIzing maps in Python}},
        year = {2019},
        DOI = {10.5281/zenodo.2609130},
        publisher = {Zenodo},
        howpublished = {\href{https://doi.org/10.5281/zenodo.2609130}{doi.org/10.5281/zenodo.2609130}}
    }
