# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# #  Exercise 01
#
# The goal of is to compare the performance of our classifier (81% accuracy) to some baseline classifiers that  would ignore the input data and instead make constant predictions.
#
# The online [documentation for DummyClassifier](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) gives instructions on how to use it.

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# %%
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_include=["int", "float"])
numerical_columns = numerical_columns_selector(data)
data_numeric = data[numerical_columns]

# %%
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# TODO: write me!
