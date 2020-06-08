# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
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
# # Exercise 02
# The goal is to find the best set of hyper-parameters which maximize the
# performance on a training set.

# %%
import numpy as np
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# TODO: create your machine learning pipeline
#
# You should:
# * preprocess the categorical columns using a `OneHotEncoder` and use a
#   `StandardScaler` to normalize the numerical data.
# * use a `LogisticRegression` as a predictive model.

# %% [markdown]
# Start by defining the columns and the preprocessing pipelines to be applied
# on each columns.
# %%

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# Subsequently, create a `ColumnTransformer` to redirect the specific columns
# a preprocessing pipeline.
# %%

from sklearn.compose import ColumnTransformer

# %% [markdown]
# Finally, concatenate the preprocessing pipeline with a logistic regression.
# %%

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


# %% [markdown]
# TODO: make your random search
#
# Use a `RandomizedSearchCV` to find the best set of hyper-parameters by tuning
# the following parameters for the `LogisticRegression` model:
# - `C` with values ranging from 0.001 to 10. You can use a reciprocal
#   distribution (i.e. `scipy.stats.reciprocal`);
# - `solver` with possible values being `"liblinear"` and `"lbfgs"`;
# - `penalty` with possible values being `"l2"` and `"l1"`;
# In addition, try several preprocessing strategies with the `OneHotEncoder`
# by always (or not) dropping the first column when encoding the categorical
# data.
#
# Notes: You can accept failure during a grid-search or a randomized-search
# by settgin `error_score` to `np.nan` for instance.
