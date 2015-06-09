SciPy 2015 Scikit-learn Tutorial
================================

Based on the SciPy [2013 tutorial](https://https://github.com/jakevdp/sklearn_scipy2013) by [Gael Varoquaux](http://gael-varoquaux.info), [Olivier Grisel](http://ogrisel.com) and [Jake VanderPlas](http://jakevdp.github.com
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
*matplotlib*, *scikit-learn*, *psutil*, *pyrallel* and *ipython* with ipython
notebook.

The last one is important, you should be able to type:

    ipython notebook

in your terminal window and see the notebook panel load in your web browser.
Because Python 3 compatibility is still being ironed-out for these packages
(we're getting close, I promise!) participants should plan to use Python 2.6
or 2.7 for this tutorial.

For users who do not yet have these  packages installed, a relatively
painless way to install all the requirements is to use a package such as
[Anaconda CE](http://store.continuum.io/ "Anaconda CE"), which can be
downloaded and installed for free.

Downloading the Tutorial Materials
----------------------------------
I would highly recommend using git, not only for this tutorial, but for the
general betterment of your life.  Once git is installed, you can clone the
material in this tutorial by using the git address shown above:

    git clone git://github.com/amueller/scipy_2015_sklearn_tutorial.git

If you can't or don't want to install git, there is a link above to download
the contents of this repository as a zip file.  I may make minor changes to
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


Detailed Outline Tutorial 1 (intermediate track)
------------------------------------------------
- 0:00 - 0:15 -- Setup and Introduction
- 0:15 - 0:30 -- Quick review of data visualization with matplotlib and numpy
- 0:30 - 1:15 -- Representation of data in machine learning
  + Downloading data within scikit-learn
  + Categorical & Image data
  + Feature extraction
- 1:15 - 2:00 -- Basic principles of Machine Learning & the scikit-learn interface
  + Supervised Learning: Classification & Regression
  + Unsupervised Learning: Clustering & Dimensionality Reduction
  + Example of PCA for data visualization
  + Flow chart: how do I choose what to do with my data set?
  + Exercise: Interactive Demo on linearly separable data
  + Regularization: what it is and why it is necessary
    - Training set vs test set error
- 2:00 - 2:15 -- Break (possibly in the middle of the previous section)
- 2:15 - 3:00 -- Supervised Learning
  + Example of Classification: hand-written digits
  + Measuring prediction performance
  + Example of Regression: boston house prices
- 3:00 - 4:15 -- Applications
  + Examples from text mining
  + Examples from image processing


Detailed Outline Tutorial 2 (advanced track)
--------------------------------------------
- 0:00 - 0:30 -- Model validation and testing
  + Bias, Variance, Over-fitting, Under-fitting
  + Using validation curves & learning  to improve your model
  + Exercise: Tuning a random forest for the digits data
- 0:30 - 1:30 -- In depth with a few learners
  + SVMs and kernels
  + Trees and forests
  + Sparse and non-sparse linear models
- 1:30 - 2:00 -- Unsupervised Learning
  + Example of Dimensionality Reduction: hand-written digits
  + Example of Clustering: Olivetti Faces
- 2:00 - 2:15 -- Pipelining learners
  + Examples of unsupervised data reduction followed by supervised learning.
- 2:15 - 2:30 -- Break (possibly in the middle of the previous section)
- 2:30 - 3:15 -- Parallel Machine Learning with IPython
  + IPython.parallel, a short primer
  + Parallel Model Assessment and Selection
  + Running a cluster on the EC2 cloud using StarCluster
- 3:15 - 4:00 -- Scaling Text Classification
  + The hashing trick
  + Online learning and out-of-core learning
  + Stochastic Gradient Descent for linear models
  + The Partition / Distribute / Average scaling scheme
