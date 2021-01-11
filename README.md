# scikit-learn course

Browse the course online: https://inria.github.io/scikit-learn-mooc

## Course description

Predictive modeling is brings value to a vast variety of data, in
business intelligence, health, industrial processes... It is a pillar of
modern data science. In this field, scikit-learn is a central tool: it is
easily accessible and yet powerful, and it dovetails in a wider ecosystem
of data-science tools based on the Python programming language.

This course is an in-depth introduction to predictive modeling with
scikit-learn. Step-by-step and didactic lessons will give you the
fundamental tools and approaches of machine learning, and is as such a
stepping stone to more advanced challenges in artificial intelligence,
text mining, or data science.

The course covers the software tools to build and evaluate predictive
pipelines, as well as the related concepts and statistical intuitions. It
is more than a cookbook: it will teach you to understand and be critical
about each step, from choosing models to interpreting them.

The course is accessible without a strong technical background, as it
only requires knowledge of basic Python programming.


## Follow the course online

A few different ways are available:
- Launch an online notebook environment using [![Binder](https://mybinder.org/badge_logo.svg)](
               https://mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)
- Browse [website](https://inria.github.io/scikit-learn-mooc) generated with
  [Jupyter Book](https://jupyterbook.org/)

## Running the notebooks locally

### Dependencies

The notebooks will require the following packages:

* python>=3.6
* jupyter
* scikit-learn
* pandas
* matplotlib
* seaborn
* plotly
* jupytext (required only for contributors)

### Local install

Materials: https://github.com/INRIA/scikit-learn-mooc/


We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

Alternatively, you can create an `scikit-learn-course` conda environment
by executing:

```
$ conda env create -f environment.yml
```

then activate the environment with:

```
$ conda activate scikit-learn-course
```

You can also update your current environment, instead of creating a new
environment, using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```

## Contributing

The source files, which should be modified, are in the `python_scripts`
directory. The notebooks are generated from these files.

### Notebooks saved in Python files

This repository uses [Jupytext](https://jupytext.readthedocs.io/) to display
Python files as notebooks. Saving as Python files facilitates version
control.

#### Setting up jupytext

When jupytext is properly connected to jupyter, the python files can be
opened in jupyter and are directly displayed as notebooks

**With jupyter notebook**

Once jupytext is installed, run the following command:

```
jupyter serverextension enable jupytext
```

**With jupyter lab**

To make it work with "jupyter lab" (instead of
"jupyter notebook"), you have to install nodejs (`conda install nodejs`
works if you use conda). Then in jupyter lab you have to right click
"Open with -> notebook" to open the python scripts with the notebook
interface.

### Updating the notebooks

To update all the notebooks:

```
$ make
```

If you want to generate a single notebook, you can do something like this:
```
$ make notebooks/02_numerical_pipeline_scaling.ipynb
```

## Direct binder links to OVH, GESIS and GKE to trigger and cache builds


- [OVH Binder](https://ovh.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)

- [GESIS Binder](https://gesis.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)

- [GKE Binder](https://gke.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)
