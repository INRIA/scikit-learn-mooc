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
# # 📝 Exercise 01
#
# The goal of this exercise is to compare the performance of our classifier
# (81% accuracy) to some baseline classifiers that would ignore the input data
# and instead make constant predictions.
#
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <= 50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
#
# Use a `sklearn.dummy.DummyClassifier` and do a train-test split to evaluate
# its accuracy on the test set. This
# [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
# shows a few examples of how to evaluate the performance of these baseline
# models.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We will first split our dataset to have the target separated from the data
# used to train our predictive model.

# %%
target_name = "class"
target = df[target_name]
data = df.drop(columns=target_name)

# %% [markdown]
# We start by selecting only the numerical columns as seen in the previous
# notebook.

# %%
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

# %% [markdown]
# Next, let's split the data and target into a train and test set.

# %%
from sklearn.model_selection import train_test_split

data_numeric_train, data_numeric_test, target_train, target_test = \
    train_test_split(data_numeric, target, random_state=0)

# %%
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Write your code here.
