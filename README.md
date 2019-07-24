Materials are from the Scipy 2018 tutorial from
(https://github.com/amueller/scipy-2018-sklearn), with an attempt to use
Jupytext for markdown files (very similar to R Markdown).

[Jupytext doc](https://jupytext.readthedocs.io/)

To generate empty notebooks from the markdown files:

```
jupytext --to ../notebooks//ipynb notebook_input_text_files/*.md
```

For generating markdown files from the notebooks:

```
jupytext --to ../markdown_files//md notebooks/*.ipynb

```

To synchronize the notebooks and the markdown files (based on filestamps, only
input cells content is modified in the notebooks):

```
jupytext --sync notebooks/*.ipynb
```

If you create a new notebook, you need to set-up the text files it is going to be paired with:
```
jupytext --set-formats notebooks//ipynb,markdown_files//md,python_scripts//py:percent notebooks/*.ipynb
```
