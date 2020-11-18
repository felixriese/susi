# -*- coding: utf-8 -*-

# Copyright (c) 2019 by Felix M. Riese. All rights reserved.

import setuptools

with open("README.rst", "r") as f:
    readme = f.read()

with open('CHANGELOG.rst', 'rb') as f:
    changelog = f.read().decode('utf-8')

long_description = '\n\n'.join((readme, changelog))

setuptools.setup(
    name="susi",
    version="1.1.1",
    author="Felix M. Riese",
    author_email="github@felixriese.de",
    description="Python package for unsupervised, supervised and semi-supervised self-organizing maps (SOM)",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/felixriese/susi",
    license="BSD-3-Clause",
    install_requires=["joblib",
                      "numpy",
                      "scikit-learn",
                      "scipy",
                      "tqdm"],
    extras_require={"docs": ["numpydoc", "sphinx", "sphinx-autobuild",
                             "sphinx_rtd_theme"]},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
