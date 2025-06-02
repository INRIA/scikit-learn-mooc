# Contributing

The source files, which should be modified, are in the `python_scripts`
directory. The notebooks are generated from these Python files with
[Jupytext](https://jupytext.readthedocs.io/).

If you are planning to contribute to this repo you should set-up your
environment using the `environment-dev.yml` or `requirements-dev.txt` file.

## Recommended workflow : Visual Studio Code

- Download [Visual Studio Code](https://code.visualstudio.com/)
- Install the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
  This will allow you to run Python files in a cell by cell fashion
- notebooks are generated through the Jupytext command-line interface (see
  [below](#updating-the-notebooks))

It may well be that alternative editors (e.g. PyCharm) allow you to have a
similar workflow, the only feature you need is to be able to run a `.py` file
in a cell-by-cell fashion with `# %%` cell markers.

### Updating the notebooks

To update all the notebooks:

```
$ make
```

If you want to generate a single notebook, you can do something similar to this:
```
$ make notebooks/02_numerical_pipeline_scaling.ipynb
```

## Alternative workflow within Jupyter

- you can configure Jupyter to open `.py` files as notebooks (see
  [below](#setting-up-jupytext) for instructions) though Jupytext
- when saving the notebook inside Jupyter it will actually write to the `.py` file

In our experience, this workflow is less convenient (Visual Studio Code is a
nicer developing environment) and also it tends to add some not very important
(and different on everyone's machine) metadata changes in the `.py` file, for
example about jupytext version, Jupyter kernel, Python version, etc ...

### Setting up jupytext

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

## Useful `make` commands to generate derived files

### Generating exercises from the solutions

```
make exercises
```

The convention in solution files:
- using `# solution` comment inside a cell will remove all the content afterwards
  (including the `# solution` line) and replace it with `# Write your code here`.
- using `tags=["solution"]` on the same line as the cell marker removes the
  whole cell, for example:
  + for Python cells: `# %% tags=["solution"]`
  + for Markdown cells: `# %% [markdown] tags=["solution"]`

### Generating quizzes questions from the gitlab quizzes

To update the github quizzes (quizzes without solutions) from the gitlab
quizzes (quizzes with solutions).

```
make quizzes
```

### Generating full-index.ipynb

full-index.ipynb is useful when giving the course live. It makes it easier to guide people
through the notebooks.

```
make full-index
```

## JupyterBook

JupyterBook is the tool we use to generate our .github.io website from our
`.py` and `.md` files (note that `.ipynb` files are not used in our JupyterBook
setup).

```
make jupyter-book
```

Generated files are in `jupyter-book/_build/html`. To open the generated JupyterBook with Firefox:
```
firefox jupyter-book/_build/html/index.html
```

Note: for slides (and maybe a few other things), you need to start a local
webserver otherwise there are issues with the way we do iframes etc ...

```sh
cd jupyter-book/_build/html/index.html && python -m http.server
```

and then open a browser at `localhost:8000`.

## Troubleshooting sphinx failures

I could not find a built-in way to pass `sphinx -P` in JupyterBook to have `pdb`
on failure. The simplest thing I found was to edit your `jupyter_book` package
(in `jupyter_book/sphinx.py`) and use:
```py
debug_args.pdb = True  # False by default
```
See https://github.com/executablebooks/jupyter-book/blob/18d52700b8636773f96631b3e187e38075557041/jupyter_book/sphinx.py#L87-L91
for the JupyterBook code.

## API diagrams

We are using app.diagrams.net to create and update some diagrams in `figures`,
notably the API ones.

To edit the diagrams:
https://app.diagrams.net/#HINRIA%2Fscikit-learn-mooc%2Fmain%2Ffigures%2Fapi_diagram.drawio

All the diagrams are in the same project, you can use the tabs at the bottom
(Google-sheet style).

Then each SVG needs to be exported individually (there could well be a better
way, update this doc if you find it): `File -> Export As -> SVG`

New modal (SVG): Click Export

New modal (Save as):
- make sure the export SVG name is correct
- click Github and select (each time sorry ...) the github project, branch and
  folder
- At the end it will ask you this file already exists do you want to override
  it: say yes
- I think it asks you to tweak the commit message if you want

Once you have done this, the SVG should have been updated in the github repo.
Make sure the github svg looks the way you want:
https://github.com/INRIA/scikit-learn-mooc/tree/main/figures

## Jupytext useful use cases

### Get wrap-up quiz solutions code

You can convert the quiz `.md` file to `.py` with valid Python code. This is
particularly useful to check quiz solutions or to try to debug user questions
on the forum.

```
jupytext --to py jupyter-book/linear_models/linear_models_wrap_up_quiz.md
```

This creates `jupyter-book/linear_models/linear_models_wrap_up_quiz.py` with
executable code.

Note: for this to work nicely with Jupytext you need to use `python` and not
`py` in the block codes i.e.:
``````
```python
```
``````

and not:
``````
```py
```
``````

##  Workflow notes

See [this](./workflow-notes.md).

## Direct binder links to OVH, GESIS and GKE to trigger and cache builds

- [OVH Binder](https://ovh.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/main?filepath=full-index.ipynb)

- [GESIS Binder](https://gesis.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/main?filepath=full-index.ipynb)

- [GKE Binder](https://gke.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/main?filepath=full-index.ipynb)

The [Binder Grafana dashboard](https://grafana.mybinder.org/d/3SpLQinmk/1-overview?orgId=1)
can be handy to have an idea whether there are currently some Binder issues.
