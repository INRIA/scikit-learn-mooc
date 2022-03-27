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
# # 📝 Exercise M1.02
#
# The goal of this exercise is to fit a similar model as in the previous
# notebook to get familiar with manipulating scikit-learn objects and in
# particular the `.fit/.predict/.score` API.

# %% [markdown]
# Let's load the adult census dataset with only numerical variables

# %%
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
data = adult_census.drop(columns="class")
target = adult_census["class"]

# %% [markdown]
# In the previous notebook we used `model = KNeighborsClassifier()`. All
# scikit-learn models can be created without arguments. This is convenient
# because it means that you don't need to understand the full details of a
# model before starting to use it.
#
# One of the `KNeighborsClassifier` parameters is `n_neighbors`. It controls
# the number of neighbors we are going to use to make a prediction for a new
# data point.
#
# What is the default value of the `n_neighbors` parameter? Hint: Look at the
# documentation on the [scikit-learn
# website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# or directly access the description inside your notebook by running the
# following cell. This will open a pager pointing to the documentation.

# %%
from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifier?

# %% [markdown]
# Create a `KNeighborsClassifier` model with `n_neighbors=50`

# %%
# Write your code here.

# %% [markdown]
# Fit this model on the data and target loaded above

# %%
# Write your code here.

# %% [markdown]
# Use your model to make predictions on the first 10 data points inside the
# data. Do they match the actual target values?

# %%
# Write your code here.

# %% [markdown]
# Compute the accuracy on the training data.

# %%
# Write your code here.

# %% [markdown]
# Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and
# compute the accuracy on the test data.

# %%
# Write your code here.
