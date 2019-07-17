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
jupytext --to ../notebooks//ipynb notebook_input_text_files/*.md

```

To synchronize the notebooks and the markdown files (based on filestamps, only
input cells content is modified in the notebooks):

```
jupytext --sync notebooks/*.ipynb
```
