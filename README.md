SciPy 2016 Scikit-learn Tutorial
================================

Based on the SciPy [2015 tutorial](https://github.com/amueller/scipy_2015_sklearn_tutorial) by [Kyle Kastner](http://https://kastnerkyle.github.io/) and [Andreas Mueller](http://amueller.github.io).


Instructors
-----------

- [Sebastian Raschka](http://sebastianraschka.com)  [@rasbt](https://twitter.com/rasbt) - Michigan State University, Computational Biology
- [Andreas Mueller](http://amuller.github.io) [@t3kcit](https://twitter.com/t3kcit) - NYU Center for Data Science


This repository will contain the teaching material and other info associated with our scikit-learn tutorial
at [SciPy 2016](http://scipy2016.scipy.org/ehome/index.php?eventid=146062&tabid=332930&) held July 11-17 in Austin, Texas.

Parts 1 to 5 make up the morning session, while
parts 6 to 9 will be presented in the afternoon.

Installation Notes
------------------

This tutorial will require recent installations of *[NumPy](http://www.numpy.org)*, *[SciPy](http://www.scipy.org)*,
*[matplotlib](http://matplotlib.org)*, *[scikit-learn](http://scikit-learn.org/stable/)* and *[IPython](http://ipython.readthedocs.org/en/stable/)* together with the *[Jupyter Notebook](http://jupyter.org)*.

The last one is important, you should be able to type:

    jupyter notebook

in your terminal window and see the notebook panel load in your web browser.
Try opening and running a notebook from the material to see check that it works.

For users who do not yet have these  packages installed, a relatively
painless way to install all the requirements is to use a Python distribution
such as [Anaconda CE](http://store.continuum.io/ "Anaconda CE"), which includes
the most relevant Python packages for science, math, engineering, and
data analysis; Anaconda can be downloaded and installed for free
including commercial use and redistribution.  
The code examples in this tutorial should be compatible to Python 2.7,
Python 3.4, and Python 3.5.

After obtaining the material, we strongly recommend you to run `python check_env.py` to verify
your environment. Although not required, we also recommend you to update the required Python packages to their latest versions to ensure best compatibility with the teaching material. Please upgrade already installed packages by executing

- `pip install [package-name] --upgrade`  
- or `conda update [package-name]`

Finally, if your environment satisfies the requirements for the tutorials, `check_env.py` will produce an output message as shown below.

    [ OK ] numpy version 1.10.4
    [ OK ] matplotlib version 1.5.1
    [ OK ] scipy version 0.17.0
    [ OK ] IPython version 4.0.3
    [ OK ] sklearn version 0.17.1
    [ OK ] pydot version 1.0.29

Downloading the Tutorial Materials
----------------------------------

We would highly recommend using git, not only for this tutorial, but for the
general betterment of your life.  Once git is installed, you can clone the
material in this tutorial by using the git address shown above:

    git clone git://github.com/amueller/scipy-2016-sklearn.git

If you can't or don't want to install git, there is a [link](https://github.com/amueller/scipy-2016-sklearn/archive/master.zip) above to download
the contents of this repository as a zip file.  We may make minor changes to
the repository in the days before the tutorial, however, so cloning the
repository is a much better option.

Data Downloads
--------------

The data for this tutorial is not included in the repository.  We will be
using several data sets during the tutorial: most are built-in to
scikit-learn, which
includes code that automatically downloads and caches these
data.  Because the wireless network
at conferences can often be spotty, it would be a good idea to download these
data sets before arriving at the conference.
Please run ``python fetch_data.py`` to download all necessary data beforehand.

Outline
=======

Coming soon
