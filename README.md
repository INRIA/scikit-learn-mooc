SciPy 2015 Scikit-learn Tutorial
================================

You can find the video recordings on youtube:

- [Part 1](https://www.youtube.com/watch?v=80fZrVMurPM)
- [Part 2](https://www.youtube.com/watch?v=Ud-FsEWegmA)


Based on the SciPy [2013 tutorial](https://github.com/jakevdp/sklearn_scipy2013) by [Gael Varoquaux](http://gael-varoquaux.info), [Olivier Grisel](http://ogrisel.com) and [Jake VanderPlas](http://jakevdp.github.com
).


Instructors
-----------
- [Kyle Kastner](https://kastnerkyle.github.io/)  [@kastnerkyle](https://twitter.com/kastnerkyle)- Université de Montréal
- [Andreas Mueller](http://amuller.github.io) [@t3kcit](https://twitter.com/t3kcit) - NYU Center for Data Science


This repository will contain files and other info associated with our Scipy
2015 scikit-learn tutorial.

Parts 1 to 5 make up the morning session, while
parts 6 to 9 will be presented in the afternoon.

Installation Notes
------------------

This tutorial will require recent installations of *numpy*, *scipy*,
*matplotlib*, *scikit-learn* and *ipython* with ipython
notebook.

The last one is important, you should be able to type:

    ipython notebook

in your terminal window and see the notebook panel load in your web browser.
Try opening and running a notebook from the material to see check that it works.

For users who do not yet have these  packages installed, a relatively
painless way to install all the requirements is to use a package such as
[Anaconda CE](http://store.continuum.io/ "Anaconda CE"), which can be
downloaded and installed for free.
Python2.7 and 3.4 should both work fine for this tutorial.

After getting the material, you should run ``python check_env.py`` to verify
your environment.

Downloading the Tutorial Materials
----------------------------------
I would highly recommend using git, not only for this tutorial, but for the
general betterment of your life.  Once git is installed, you can clone the
material in this tutorial by using the git address shown above:

    git clone git://github.com/amueller/scipy_2015_sklearn_tutorial.git

If you can't or don't want to install git, there is a link above to download
the contents of this repository as a zip file.  We may make minor changes to
the repository in the days before the tutorial, however, so cloning the
repository is a much better option.

Data Downloads
--------------
The data for this tutorial is not included in the repository.  We will be
using several data sets during the tutorial: most are built-in to
scikit-learn, which
includes code which automatically downloads and caches these
data.  Because the wireless network
at conferences can often be spotty, it would be a good idea to download these
data sets before arriving at the conference.
Run ``fetch_data.py`` to download all necessary data beforehand.

Outline
=======

Morning Session
----------------
- What is machine learning? (Sample applications)
- Kinds of machine learning: unsupervised vs supervised.
- Data formats and preparation.
- Supervised learning
    - Interface
    - Training and test data
    - Classification
    - Regression
- Unsupervised Learning
    - Unsupervised transformers
    - Preprocessing and scaling
    - Dimensionality reduction
    - Clustering
- Summary : Estimator interface
- Application : Classification of digits
- Application : Eigenfaces
- Methods: Text feature abstraction, bag of words
- Application : SMS spam detection
- Summary : Model building and generalization


Afternoon Session
------------------
- Cross-Validation
- Model Complexity: Overfitting and underfitting
- Complexity of various model types
- Grid search for adjusting hyperparameters 
- Basic regression with cross-validation
- Application : Titanic survival with Random Forest
- Building Pipelines
    - Motivation and Basics
    - Preprocessing and Classification
    - Grid-searching Parameters of the feature extraction
- Application : Image classification
- Model complexity, learning curves and validation curves
- In-Depth supervised models
    - Linear Models
    - Kernel SVMs
    - trees and Forests
- Learning with Big Data
    - Out-Of-Core learning
    - The hashing trick for large text corpuses
