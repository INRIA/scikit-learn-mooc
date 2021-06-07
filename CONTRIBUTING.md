# Contributing

The source files, which should be modified, are in the `python_scripts`
directory. The notebooks are generated from these files.

## Notebooks saved in Python files

This repository uses [Jupytext](https://jupytext.readthedocs.io/) to display
Python files as notebooks. Saving as Python files facilitates version
control.

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

## Updating the notebooks

To update all the notebooks:

```
$ make
```

If you want to generate a single notebook, you can do something like this:
```
$ make notebooks/02_numerical_pipeline_scaling.ipynb
```

## Troubleshooting sphinx failures
 
I could not find a built-in way to pass `sphinx -P` in JupyterBook to have `pdb`
on failure. The simplest thing I found was to edit your jupyter_book package
(in jupyter_book/sphinx.py) and use: 
```py
debug_args.pdb = True  # False by default
```
See https://github.com/executablebooks/jupyter-book/blob/18d52700b8636773f96631b3e187e38075557041/jupyter_book/sphinx.py#L87-L91
for the JupyterBook code.

## API diagrams

We are using app.diagrams.net to create and update some diagrams in `figures`,
notably the API ones.

To edit the diagrams:
https://app.diagrams.net/#HINRIA%2Fscikit-learn-mooc%2Fmaster%2Ffigures%2Fapi_diagram.drawio

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
https://github.com/INRIA/scikit-learn-mooc/tree/master/figures

## Direct binder links to OVH, GESIS and GKE to trigger and cache builds


- [OVH Binder](https://ovh.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)

- [GESIS Binder](https://gesis.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)

- [GKE Binder](https://gke.mybinder.org/v2/gh/INRIA/scikit-learn-mooc/master)
