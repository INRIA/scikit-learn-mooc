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
# # 📃 Solution for Exercise M1.02
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
# scikit-learn models can be created without arguments, which means that you
# don't need to understand the details of the model to use it in scikit-learn.
#
# One of the `KNeighborsClassifier` parameters is `n_neighbors`. It controls
# the number of neighbors we are going to use to make a prediction for a new
# data point.
#
# What is the default value of the `n_neighbors` parameter? Hint: Look at the
# help inside your notebook `KNeighborsClassifier?` or on the [scikit-learn
# website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

# %% [markdown] tags=["solution"]
# The default value for `n_neighbors` is 5

# %% [markdown]
# Create a `KNeighborsClassifier` model with `n_neighbors=50`

# %%
# solution
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=50)

# %% [markdown]
# Fit this model on the data and target loaded above

# %%
# solution
model.fit(data, target)

# %% [markdown]
# Use your model to make predictions on the first 10 data points inside the
# data. Do they match the actual target values?

# %%
# solution
first_data_values = data.iloc[:10]
first_predictions = model.predict(first_data_values)
first_predictions

# %% tags=["solution"]
first_target_values = target.iloc[:10]
first_target_values

# %% tags=["solution"]
number_of_correct_predictions = (
    first_predictions == first_target_values).sum()
number_of_predictions = len(first_predictions)
print(
    f"{number_of_correct_predictions}/{number_of_predictions} "
    "of predictions are correct")

# %% [markdown]
# Compute the accuracy on the training data.

# %%
# solution
model.score(data, target)

# %% [markdown]
# Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and
# compute the accuracy on the test data.

# %%
# solution
adult_census_test = pd.read_csv("../datasets/adult-census-numeric-test.csv")

data_test = adult_census_test.drop(columns="class")
target_test = adult_census_test["class"]

model.score(data_test, target_test)

# %% [markdown] tags=["solution"]
# Looking at the previous notebook, the accuracy seems slightly higher with
# `n_neighbors=50` than with `n_neighbors=5` (the default value).
