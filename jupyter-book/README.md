Install the dependencies for jupyter-book:
```sh
pip install myst-parser myst-nb jupyter-book
```

```sh
cd jupyter-book
```

Create the .md files from the .py files (at the time of writing .py files
are not supported in jupyter-book):
```sh
bash generate-md-files.sh
```

To generate the jupyter-book:
```sh
jb build .
```

Generated file are in `__build/html`. For example you can do:
```sh
firefox _build/html/index.html
```
