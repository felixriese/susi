---
home: true
heroImage: https://github.com/felixriese/susi/blob/main/docs/_static/susi_logo.png?raw=true
heroText: SuSi – Self-organizing Maps with Python
tagline: "Self-organizing maps (SOM) Python package for unsupervised, supervised and semi-supervised learning"
actionText: Get Started →
actionLink: https://susi.readthedocs.io/en/latest/install.html
features:
- title: Simple syntax
  details: You can use SuSi similar to the well-known scikit-learn syntax, e.g., fit(), predict(), transform().
- title: Multiple applications
  details: Support for unsupervised clustering and visualization, semi-supervised and supervised classification and regression.
- title: Built-in visualization
  details: Present your data with SuSi through built-in plotting scripts and example notebooks.
footer: Copyright © 2019-2022 Felix M. Riese
# search: false

---

### What is SuSi?

**SuSi** is a **Python package** for unsupervised, supervised, and semi-supervised learning. It is built as an estimator in [**scikit-learn**](https://scikit-learn.org) style and works with all currently-maintained [**Python 3**](https://python.org) versions.

This is a basic example on how to use **SuSi** for supervised classification:

```python
import susi

# load your dataset
X_train, X_test, y_train, y_test = ...

# initialize and fit SuSi
som = susi.SOMClassifier()
som.fit(X_train, y_train)

# predict and calculate the accuracy score
y_pred = som.predict(X_test)
print(som.score(X_test, y_test))
```

### Getting started

Installation of **SuSi** via pip:

```bash
pip install susi
```

Installation of **SuSi** via conda-forge:

```bash
conda install -c conda-forge susi
```
