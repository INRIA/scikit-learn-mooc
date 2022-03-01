# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M3.02
#
# The goal is to find the best set of hyperparameters which maximize the
# generalization performance on a training set.
#
# Here again with limit the size of the training set to make computation
# run faster. Feel free to increase the `train_size` value if your computer
# is powerful enough.

# %%

import numpy as np
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)

# %% [markdown]
# In this exercise, we will progressively define the classification pipeline
# and later tune its hyperparameters.
#
# Our pipeline should:
# * preprocess the categorical columns using a `OneHotEncoder` and use a
#   `StandardScaler` to normalize the numerical data.
# * use a `LogisticRegression` as a predictive model.
#
# Start by defining the columns and the preprocessing pipelines to be applied
# on each group of columns.
# %%
from sklearn.compose import make_column_selector as selector

# Write your code here.

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Write your code here.

# %% [markdown]
# Subsequently, create a `ColumnTransformer` to redirect the specific columns
# a preprocessing pipeline.

# %%
from sklearn.compose import ColumnTransformer

# Write your code here.

# %% [markdown]
# Assemble the final pipeline by combining the above preprocessor
# with a logistic regression classifier. Force the maximum number of
# iterations to `10_000` to ensure that the model will converge.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Write your code here.

# %% [markdown]
# Use `RandomizedSearchCV` with `n_iter=20` to find the best set of
# hyperparameters by tuning the following parameters of the `model`:
#
# - the parameter `C` of the `LogisticRegression` with values ranging from
#   0.001 to 10. You can use a log-uniform distribution
#   (i.e. `scipy.stats.loguniform`);
# - the parameter `with_mean` of the `StandardScaler` with possible values
#   `True` or `False`;
# - the parameter `with_std` of the `StandardScaler` with possible values
#   `True` or `False`.
#
# Once the computation has completed, print the best combination of parameters
# stored in the `best_params_` attribute.

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# Write your code here.
