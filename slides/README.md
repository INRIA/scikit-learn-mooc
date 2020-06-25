# Preview

To preview slides `https://inria.github.io/scikit-learn-mooc/slides/?file=[FILENAME].md`

Example : https://inria.github.io/scikit-learn-mooc/slides/?file=ml_concepts.md

# Export

To install packages needed to generate the slides:

```
pip install -r requirements.txt
```

Note: for some reason if you `pip install` `htmlark` and not
`htmlark[parsers]`, you'll get a blank HTML page.

Then use `make` to export html files.
