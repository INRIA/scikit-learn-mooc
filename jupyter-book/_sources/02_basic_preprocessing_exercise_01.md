---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

#  Exercise 01

The goal of this exercise is to compare the performance of our classifier
(81% accuracy) to some baseline classifiers that would ignore the input data
and instead make constant predictions.

- What would be the score of a model that always predicts `' >50K'`?
- What would be the score of a model that always predicts `' <= 50K'`?
- Is 81% or 82% accuracy a good score for this problem?

Use a `DummyClassifier` and do a train-test split to evaluate
its accuracy on the test set. This
[link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
shows a few examples of how to evaluate the performance of these baseline
models.

```{code-cell}
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")
```

```{code-cell}
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])
```

```{code-cell}
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_include=["int", "float"])
numerical_columns = numerical_columns_selector(data)
data_numeric = data[numerical_columns]
```

```{code-cell}
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# TODO: write me!
```
