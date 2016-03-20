SciPy 2015 Scikit-learn Tutorial
================================

Based on the SciPy [2015 tutorial](https://github.com/amueller/sklearn_scipy2013) by [Gael Varoquaux](http://gael-varoquaux.info), [Olivier Grisel](http://ogrisel.com) and [Jake VanderPlas](http://jakevdp.github.com
).


Instructors
-----------
- [Sebastian Raschka](https://http://sebastianraschka.com/)  [@rasbt](https://twitter.com/rasbt)
- [Andreas Mueller](http://amuller.github.io) [@t3kcit](https://twitter.com/t3kcit) - NYU Center for Data Science


This repository will contain files and other info associated with our Scipy
2016 scikit-learn tutorial.

Parts 1 to 5 make up the morning session, while
parts 6 to 9 will be presented in the afternoon.

Installation Notes
------------------

This tutorial will require recent installations of *numpy*, *scipy*,
*matplotlib*, *scikit-learn* and *ipython* together with the *jupyter notebook*

The last one is important, you should be able to type:

    jupyter notebook

in your terminal window and see the notebook panel load in your web browser.
Try opening and running a notebook from the material to see check that it works.

For users who do not yet have these  packages installed, a relatively
painless way to install all the requirements is to use a package such as
[Anaconda CE](http://store.continuum.io/ "Anaconda CE"), which can be
downloaded and installed for free.
Python2.7 Python3.4 and Python3.5 should all work for this tutorial.

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

To come
