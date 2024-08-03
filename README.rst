.. image:: https://badge.fury.io/py/susi.svg
    :target: https://pypi.org/project/susi/
    :alt: PyPi - Code Version

.. image:: https://img.shields.io/pypi/pyversions/susi.svg
    :target: https://pypi.org/project/susi/
    :alt: PyPI - Python Version

.. image:: https://readthedocs.org/projects/susi/badge/?version=latest
    :target: https://susi.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/felixriese/susi/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/felixriese/susi
    :alt: Codecov

.. image:: https://api.codacy.com/project/badge/Grade/d304689a7364437db1ef998cf7765f5a
	:target: https://app.codacy.com/app/felixriese/susi
	:alt: Codacy Badge

.. image:: https://anaconda.org/conda-forge/susi/badges/version.svg
    :target: https://anaconda.org/conda-forge/susi
    :alt: Conda-forge

|

.. image:: https://raw.githubusercontent.com/felixriese/susi/master/docs/_static/susi_logo_small.png
    :target: https://github.com/felixriese/susi
    :align: right
    :alt: SuSi logo

SuSi: Supervised Self-organizing maps in Python
===============================================

Python package for unsupervised, supervised and semi-supervised self-organizing maps (SOM)

Description
-----------

We present the SuSi package for Python.
It includes a fully functional SOM for unsupervised, supervised and semi-supervised tasks:

- SOMClustering: Unsupervised SOM for clustering
- SOMRegressor: (Semi-)Supervised Regression SOM
- SOMClassifier: (Semi-)Supervised Classification SOM

:License:
    `3-Clause BSD license <LICENSE>`_

:Author:
    `Felix M. Riese <mailto:github@felixriese.de>`_

:Citation:
    see `Citation`_ and in the `bibtex <https://github.com/felixriese/susi/blob/main/bibliography.bib>`_ file

:Documentation:
    `Documentation <https://susi.readthedocs.io/en/latest/index.html>`_

:Installation:
    `Installation guidelines <https://susi.readthedocs.io/en/latest/install.html>`_

:Paper:
    `F. M. Riese, S. Keller and S. Hinz in Remote Sensing, 2020 <https://www.mdpi.com/2072-4292/12/1/7>`_


Installation
------------

Pip
~~~

.. code:: bash

    pip3 install susi
    
.. image:: https://static.pepy.tech/personalized-badge/susi?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads
	:target: https://pepy.tech/project/susi
	:alt: PyPi Downloads

Conda
~~~~~

.. code:: bash

    conda install -c conda-forge susi

More information can be found in the `installation guidelines <https://susi.readthedocs.io/en/latest/install.html>`_.

.. image:: https://img.shields.io/conda/dn/conda-forge/susi.svg
	:target: https://anaconda.org/conda-forge/susi
	:alt: Conda-Forge Downloads

Examples
--------

A collection of code examples can be found in `the documentation <https://susi.readthedocs.io/en/latest/examples.html>`_.
Code examples as Jupyter Notebooks can be found here:

* `examples/SOMClustering <https://github.com/felixriese/susi/blob/main/examples/SOMClustering.ipynb>`_
* `examples/SOMRegressor <https://github.com/felixriese/susi/blob/main/examples/SOMRegressor.ipynb>`_
* `examples/SOMRegressor_semisupervised <https://github.com/felixriese/susi/blob/main/examples/SOMRegressor_semisupervised.ipynb>`_
* `examples/SOMRegressor_multioutput <https://github.com/felixriese/susi/blob/main/examples/SOMRegressor_multioutput.ipynb>`_
* `examples/SOMClassifier <https://github.com/felixriese/susi/blob/main/examples/SOMClassifier.ipynb>`_
* `examples/SOMClassifier_semisupervised <https://github.com/felixriese/susi/blob/main/examples/SOMClassifier_semisupervised.ipynb>`_

FAQs
-----

- **How should I set the initial hyperparameters of a SOM?** For more details
  on the hyperparameters, see in `documentation/hyperparameters
  <https://susi.readthedocs.io/en/latest/hyperparameters.html>`_.
- **How can I optimize the hyperparameters?** The SuSi hyperparameters
  can be optimized, for example, with `scikit-learn.model_selection.GridSearchCV
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_,
  since the SuSi package is developed according to several scikit-learn
  guidelines.


------------


Citation
--------

The bibtex file including both references is available in `bibliography.bib
<https://github.com/felixriese/susi/blob/main/bibliography.bib>`_.

**Paper:**

F. M. Riese, S. Keller and S. Hinz, "Supervised and Semi-Supervised Self-Organizing
Maps for Regression and Classification Focusing on Hyperspectral Data",
*Remote Sensing*, vol. 12, no. 1, 2020. `DOI:10.3390/rs12010007
<https://doi.org/10.3390/rs12010007>`_

.. code:: bibtex

    @article{riese2020supervised,
        author = {Riese, Felix~M. and Keller, Sina and Hinz, Stefan},
        title = {{Supervised and Semi-Supervised Self-Organizing Maps for
                  Regression and Classification Focusing on Hyperspectral Data}},
        journal = {Remote Sensing},
        year = {2020},
        volume = {12},
        number = {1},
        article-number = {7},
        URL = {https://www.mdpi.com/2072-4292/12/1/7},
        ISSN = {2072-4292},
        DOI = {10.3390/rs12010007}
    }

**Code:**

Felix M. Riese, "SuSi: SUpervised Self-organIzing maps in Python",
Zenodo, 2019. `DOI:10.5281/zenodo.2609130
<https://doi.org/10.5281/zenodo.2609130>`_

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2609130.svg
   :target: https://doi.org/10.5281/zenodo.2609130

.. code:: bibtex

    @misc{riese2019susicode,
        author = {Riese, Felix~M.},
        title = {{SuSi: Supervised Self-Organizing Maps in Python}},
        year = {2019},
        DOI = {10.5281/zenodo.2609130},
        publisher = {Zenodo},
        howpublished = {\href{https://doi.org/10.5281/zenodo.2609130}{doi.org/10.5281/zenodo.2609130}}
    }

-------------

License
-------

This project is published under the `3-Clause BSD <LICENSE>`_ license.

.. image:: https://img.shields.io/pypi/l/susi.svg
    :target: https://github.com/felixriese/susi/blob/main/LICENSE
    :alt: PyPI - License
