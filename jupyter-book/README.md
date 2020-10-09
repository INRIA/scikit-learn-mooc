Install the dependencies for jupyter-book:
```sh
pip install myst-parser myst-nb jupyter-book
```

```sh
cd jupyter-book
```

To generate the jupyter-book HTML:
```sh
jb build .
```

Generated file are in `__build/html`. For example you can do:
```sh
firefox _build/html/index.html
```
