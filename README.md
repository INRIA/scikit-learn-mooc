# scikit-learn course

Browse the course online: https://inria.github.io/scikit-learn-mooc

## Course description

The course description can be found here:
https://inria.github.io/scikit-learn-mooc/index.html


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

see instructions [here](./local-install-instructions.md)

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
