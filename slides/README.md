# View slides

## On the .github.io website

The general pattern is `https://inria.github.io/scikit-learn-mooc/slides/?file=[FILENAME].md`

Example for ML concepts slides:
https://inria.github.io/scikit-learn-mooc/slides/?file=ml_concepts.md

## Locally

Useful when working on the slides:

```py
# on the root repo folder
python -m http.server

# open your browser with the right port (from previous command) using the right md file
firefox 'http://localhost:8000/slides/index.html?file=../slides/ml_concepts.md'
```

# Export

To install packages needed to generate the slides:

```
pip install -r requirements.txt
```

Note: for some reason if you `pip install` `htmlark` and not
`htmlark[parsers]`, you'll get a blank HTML page.

Then use `make` to export html files.
