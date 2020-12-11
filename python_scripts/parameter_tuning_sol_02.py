# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise 01
#
# The goal is to write an exhaustive search to find the best parameters
# combination maximizing the model performance.
#
# Here we use a small subset of the Adult Census dataset to make to code
# fast to execute. Once your code works on the small subset, try to
# change `train_size` to a larger value (e.g. 0.8 for 80% instead of
# 20%).

# %%
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = df[target_name]
data = df.drop(columns=[target_name, "fnlwgt"])

df_train, df_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'native-country', 'sex']

categories = [data[column].unique()
              for column in data[categorical_columns]]

categorical_preprocessor = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer(
    [('cat-preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# This line is currently required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])

# %% [markdown]
#
# Use the previously defined model (called `model`) and using two nested `for`
# loops, make a search of the best combinations of the `learning_rate` and
# `max_leaf_nodes` parameters. In this regard, you will need to train and test
# the model by setting the parameters. The evaluation of the model should be
# performed using `cross_val_score`. We will use the following parameters
# search:
# - `learning_rate` for the values 0.01, 0.1, 1 and 10. This parameter controls
#   the ability of a new tree to correct the error of the previous sequence of
#   trees
# - `max_leaf_nodes` for the values 3, 10, 30. This parameter controls the
#   depth of each tree.

# %%
from sklearn.model_selection import cross_val_score

learning_rate = [0.05, 0.1, 0.5, 1, 5]
max_leaf_nodes = [3, 10, 30, 100]

best_score = 0
best_params = {}
for lr in learning_rate:
    for mln in max_leaf_nodes:
        print(f"Evaluating model with learning rate {lr:.3f}"
              f" and max leaf nodes {mln}... ", end="")
        model.set_params(
            classifier__learning_rate=lr,
            classifier__max_leaf_nodes=mln
        )
        scores = cross_val_score(model, df_train, target_train, cv=2)
        mean_score = scores.mean()
        print(f"score: {mean_score:.3f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'learning-rate': lr, 'max leaf nodes': mln}
            print(f"Found new best model with score {best_score:.3f}!")

print(f"The best accuracy obtained is {best_score:.3f}")
print(f"The best parameters found are:\n {best_params}")
