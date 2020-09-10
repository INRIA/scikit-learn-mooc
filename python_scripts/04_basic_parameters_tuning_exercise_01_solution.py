# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# #  Exercise 01
# The goal is to write an exhaustive search to find the best parameters
# combination maximizing the model performance

# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# This line is currently required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from scipy.stats import expon, uniform
from scipy.stats import randint

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

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

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    preprocessor, HistGradientBoostingClassifier(random_state=42))

# %% [markdown]
# TODO: write your solution here
#
# Use the previously defined model (called `model`) and using two nested `for`
# loops, make a search of the best combinations of the `learning_rate` and
# `max_leaf_nodes` parameters. In this regard, you will need to train and test
# the model by setting the parameters. The evaluation of the model should be
# performed using `cross_val_score`. We can propose to define the following
# parameters search:
# - `learning_rate` for the values 0.01, 0.1, and 1;
# - `max_leaf_nodes` for the values 5, 25, 45.

# %%
from sklearn.model_selection import cross_val_score

learning_rate = [0.01, 0.1, 1, 10]
max_leaf_nodes = [5, 25, 45]

best_score = 0
best_params = {}
for lr in learning_rate:
    for mln in max_leaf_nodes:
        model.set_params(
            histgradientboostingclassifier__learning_rate=lr,
            histgradientboostingclassifier__max_leaf_nodes=mln
        )
        scores = cross_val_score(model, df_train, target_train, cv=3)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_params = {'learning-rate': lr, 'max leaf nodes': mln}
print(f"The best accuracy obtained is {best_score:.3f}")
print(f"The best parameters found are:\n {best_params}")
