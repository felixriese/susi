# -*- coding: utf-8 -*-

import setuptools

with open("README.rst", "r") as f:
    readme = f.read()

with open("CHANGELOG.rst", "r") as f:
    changelog = f.read()

long_description = "\n\n".join((readme, changelog))

setuptools.setup(
    name="susi",
    version="1.4.2",
    author="Felix M. Riese",
    author_email="github@felixriese.de",
    description=(
        "Python package for unsupervised, supervised and "
        "semi-supervised self-organizing maps (SOM)"
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/felixriese/susi",
    license="BSD-3-Clause",
    install_requires=[
        "joblib",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        "docs": ["numpydoc", "sphinx", "sphinx-autobuild", "sphinx_rtd_theme"],
        "examples": ["notebook", "seaborn", "pandas"],
        "tests": ["pytest", "pytest-cov", "codecov", "nbval", "coverage"],
        "formatting": ["black", "isort"],
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={
        "Documentation": "https://susi.readthedocs.io/en/latest/?badge=latest",
        "Source": "https://github.com/felixriese/susi",
        "Tracker": "https://github.com/felixriese/susi/issues",
    },
    python_requires=">=3.9",
)
