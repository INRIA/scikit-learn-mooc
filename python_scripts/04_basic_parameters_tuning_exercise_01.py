# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# #  Exercise 01
# The goal is to find the best set of hyper-parameters which maximize the
# performance on a training set.

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

df = pd.read_csv("https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
# TODO: write your solution here
# You should:
# - create a preprocessor using an `OrdinalEncoder`
# - use a `HistGradientBoostingClassifier` to make predictions
# - use a `RandomizedSearchCV` to find the best set of hyper-parameters by
#   tuning the following parameters: `learning_rate`, `l2_regularization`,
#   `max_leaf_nodes`, and `min_samples_leaf`.
