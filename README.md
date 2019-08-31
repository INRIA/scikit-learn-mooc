# EuroSciPy 2019 - scikit-learn tutorial

## Getting started

The tutorials will require the following packages:

* python>=3.6
* jupyter
* scikit-learn
* pandas
* pandas-profiling
* matplotlib
* seaborn

We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

You can create an `sklearn-tutorial` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate sklearn-tutorial
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```

## Contributing

This repo uses: [Jupytext doc](https://jupytext.readthedocs.io/)

To synchronize the notebooks and the Python scripts (based on filestamps, only
input cells content is modified in the notebooks):

```
$ jupytext --sync notebooks/*.ipynb
```

or simply use:

```
$ make sync
```

If you create a new notebook, you need to set-up the text files it is going to
be paired with:

```
$ jupytext --set-formats notebooks//ipynb,python_scripts//py:percent notebooks/*.ipynb
```

or simply use:

```
$ make format
```

To render all the notebooks (from time to time, slow to run):

```
$ make render
```
