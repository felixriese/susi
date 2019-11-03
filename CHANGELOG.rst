Change Log
==========

[1.0.7] - coming soon
------------------------
- [ADDED] Optional tqdm visualization of the SOM training
- [CHANGED] Docstrings for functions are now according to guidelines.
- [FIXED] Semi-supervised classification handling
- [FIXED] Supervised classification SOM initalization of `n_iter_supervised`
- [FIXED] Code refactored according to prospector

[1.0.6] - 2019-09-11
------------------------
- [ADDED] Semi-supervised abilities for classifier and regressor
- [ADDED] Example notebooks for semi-supervised applications
- [ADDED] Tests for example notebooks
- [CHANGED] Requirements for the SUSI package
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
