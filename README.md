# scikit-learn tutorial

All notebook material: https://github.com/INRIA/scikit-learn-mooc/

## Follow the tutorial online

A few different ways are available:
- Launch an online notebook environment using [![Binder](https://mybinder.org/badge_logo.svg)](
               https://mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)
- Browse [website](https://inria.github.io/scikit-learn-mooc/jupyter-book/) generated with
  [Jupyter Book](https://jupyterbook.org/)

## Running the tutorial locally

### Dependencies

The tutorials will require the following packages:

* python>=3.6
* jupyter
* scikit-learn
* pandas
* matplotlib
* seaborn
* plotly
* jupytext (required only for contributors)

### Local install

We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

Alternatively, you can create an `scikit-learn-tutorial` conda environment
by executing:

```
$ conda env create -f environment.yml
```

then activate the environment with:

```
$ conda activate scikit-learn-tutorial
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

In some cases you
may need to use a `jupytext` command directly rather than using the provided
`Makefile`. For instance, to create a notebook from a Python script:
```
$ jupytext --to ipynb python_scripts/your_python_script.py --output notebooks/your_notebook.ipynb
```

## Direct binder links to OVH, GESIS and GKE to trigger and cache builds


- [OVH Binder](https://ovh.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)

- [GESIS Binder](https://gesis.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)

- [GKE Binder](https://gke.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)
