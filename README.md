This repo uses: [Jupytext doc](https://jupytext.readthedocs.io/)

To synchronize the notebooks and the Python scripts (based on filestamps, only
input cells content is modified in the notebooks):

```
jupytext --sync notebooks/*.ipynb
```

If you create a new notebook, you need to set-up the text files it is going to be paired with:

```
jupytext --set-formats notebooks//ipynb,python_scripts//py:percent notebooks/*.ipynb
```

To render all the notebooks (from time to time, slow to run):

```
make render
```
