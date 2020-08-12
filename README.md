# scikit-learn tutorial

All notebook material: https://github.com/INRIA/scikit-learn-mooc/

## Follow the tutorial online

- Launch an online notebook environment using [![Binder](https://mybinder.org/badge_logo.svg)](
               https://mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master?urlpath=lab)

- Browse the static content online (pre-rendered outputs) using [nbviewer](
  https://nbviewer.jupyter.org/github/INRIA/scikit-learn-mooc/tree/master/rendered_notebooks/)

You will need an internet connection but you do not have to install any packages
locally.


## Running the tutorial locally

### Dependencies

The tutorials will require the following packages:

* python>=3.6
* jupyter
* scikit-learn
* pandas
* pandas-profiling
* matplotlib
* seaborn

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

### Building the rendered notebooks


To rebuild all the rendered notebooks (from time to time, slow to run):

```
$ make
```

In some cases you
may need to use a `jupytext` command directly rather than using the provided
`Makefile`. For instance, to create an empty notebook from a Python script:
```
$ jupytext --to ../notebooks//ipynb python_scripts/your_python_script.py
```

## Direct binder links to GKE and OVH to trigger and cache builds

- [GKE Binder](https://gke.mybinder.org/v2/gh/lesteve/scikit-learn-tutorial/master?urlpath=lab)

- [OVH Binder](https://ovh.mybinder.org/v2/gh/lesteve/scikit-learn-tutorial/master?urlpath=lab)
