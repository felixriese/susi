Change Log
==========

[1.1.1] - 2020-11-18
--------------------
- [ADDED] New distance metric "spectralangle".
- [ADDED] FAQs.
- [ADDED] Separate between positional and keyword parameters.
- [ADDED] Plot script for neighborhood distance weight matrix.
- [FIXED] Added inherited members to code documentation.

[1.1.0] - 2020-08-31
--------------------
- [ADDED] Logo.
- [ADDED] SOMPlots documentation.
- [REMOVED] Python 3.5 support. Now, only 3.6-3.8 are supported.
- [FIXED] Scikit-learn warnings regarding validation of positional arguments.
- [FIXED] Sphinx documentation warnings.

[1.0.10] - 2020-04-21
------------------------------------
- [ADDED] Support for Python 3.8.x.
- [ADDED] Test coverage and MultiOutput test.
- [CHANGED] Function `setPlaceholder` to `set_placeholder`.
- [FIXED] Documentation links

[1.0.9] - 2020-04-07
------------------------
- [ADDED] Documentation of the hyperparameters.
- [ADDED] Plot scripts.
- [CHANGED] Structure of the module files.

[1.0.8] - 2020-01-20
------------------------
- [FIXED] Replaced scikit-learn `sklearn.utils.fixes.parallel_helper`, see #12.

[1.0.7] - 2019-11-28
------------------------
- [ADDED] Optional tqdm visualization of the SOM training
- [ADDED] New `init_mode_supervised` called `random_minmax`.
- [CHANGED] Official name of package changes from `SUSI` to `SuSi`.
- [CHANGED] Docstrings for functions are now according to guidelines.
- [FIXED] Semi-supervised classification handling, sample weights
- [FIXED] Supervised classification SOM initalization of `n_iter_supervised`
- [FIXED] Code refactored according to prospector
- [FIXED] Resolved bug in `get_datapoints_from_node()` for unsupervised SOM.

[1.0.6] - 2019-09-11
------------------------
- [ADDED] Semi-supervised abilities for classifier and regressor
- [ADDED] Example notebooks for semi-supervised applications
- [ADDED] Tests for example notebooks
- [CHANGED] Requirements for the SuSi package
- [REMOVED] Support for Python 3.4
- [FIXED] Code looks better in documentation with sphinx.ext.napoleon

[1.0.5] - 2019-04-23
------------------------
- [ADDED] PCA initialization of the SOM weights with 2 principal components
- [ADDED] Variable variance
- [CHANGED] Moved installation guidelines and examples to documentation

[1.0.4] - 2019-04-21
------------------------
- [ADDED] Batch algorithm for unsupervised and supervised SOM
- [ADDED] Calculation of the unified distance matrix (u-matrix)
- [FIXED] Added estimator_check of scikit-learn and fixed recognized issues

[1.0.3] - 2019-04-09
------------------------
- [ADDED] Link to arXiv paper
- [ADDED] Mexican-hat neighborhood distance weight
- [ADDED] Possibility for different initialization modes
- [CHANGED] Simplified initialization of estimators
- [FIXED] URLs and styles in documentation
- [FIXED] Colormap in Salinas example

[1.0.2] - 2019-03-27
------------------------
- [ADDED] Codecov, Codacy
- [CHANGED] Moved decreasing_rate() out of SOM classes
- [FIXED] Removed duplicate constructor for SOMRegressor, fixed fit() params

[1.0.1] - 2019-03-26
------------------------
- [ADDED] Config file for Travis
- [ADDED] Requirements for read-the-docs documentation

[1.0.0] - 2019-03-26
------------------------
- Initial release
