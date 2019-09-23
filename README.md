# scikit-learn tutorial

All notebook material: https://github.com/lesteve/scikit-learn-tutorial/

## Follow the tutorial online

- Launch an online notebook environment using [![Binder](https://mybinder.org/badge_logo.svg)](
               https://mybinder.org/v2/gh/lesteve/scikit-learn-tutorial/master?urlpath=lab)

- Browse the static content online (pre-rendered outputs) using [nbviewer](
  https://nbviewer.jupyter.org/github/lesteve/scikit-learn-tutorial/tree/master/rendered_notebooks/)

You need an internet connection but you will not have to install any package
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

You can create an `scikit-learn-tutorial` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate scikit-learn-tutorial
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```

## Contributing

To synchronize the notebooks and the Python scripts (based on filestamps, only
input cells content is modified in the notebooks):

```
$ make notebooks
```

To render all the notebooks (from time to time, slow to run):

```
$ make
```

This repo uses [Jupytext](https://jupytext.readthedocs.io/). In some cases you
may need to use a `jupytext` command directly rather than using the provided
`Makefile`. Here are a few useful `jupytext` commands:
- pair a notebook with a Python script:
```
$ jupytext --set-formats python_scripts//py:percent,notebooks//ipynb notebooks/your_notebook.ipynb
```
- sync a paired Python script and notebook:
```
$ jupytext --sync notebooks/your_notebook.ipynb
```
- create an empty notebook from a Python script:
```
$ jupytext --to ../notebooks//ipynb python_scripts/your_python_script.py
```

## Direct binder links to GKE and OVH to trigger and cache builds

- [GKE Binder](https://gke.mybinder.org/v2/gh/lesteve/scikit-learn-tutorial/master?urlpath=lab)

- [OVH Binder](https://ovh.mybinder.org/v2/gh/lesteve/scikit-learn-tutorial/master?urlpath=lab)
