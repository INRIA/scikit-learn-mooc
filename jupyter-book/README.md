Install the dependencies for jupyter-book:
```sh
pip install myst-parser myst-nb jupyter-book
```

```sh
cd jupyter-book
```

To generate the jupyter-book HTML:
```sh
make
```

Generated files are in `_build/html`. To browse the generated website it is
recommended to use a local web server:

```sh
cd _build/html/index.html && python -m http.server
```

and then open a browser at `localhost:8000`. A few things will not work, e.g.
slides, if you open the HTML files in a web browser directly.

To generate the quizzes without solution from our private repo:
```sh
cd jupyter-book

make quizzes
```
