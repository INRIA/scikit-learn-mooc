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
# # Exercise 02
#
# The goal of this exercise is to evalutate the impact of using an arbitrary
# integer encoding for categorical variables along with a linear
# classification model such as Logistic Regression.
#
# To do so, let's try to use `OrdinalEncoder` to preprocess the categorical
# variables. This preprocessor is assembled in a pipeline with
# `LogisticRegression`. The performance of the pipeline can be evaluated as
# usual by cross-validation and then compared to the score obtained when using
# `OneHotEncoding` or to some other baseline score.
#
# Because `OrdinalEncoder` can raise errors if it sees an unknown category at
# prediction time, we need to pre-compute the list of all possible categories
# ahead of time:
#
# ```python
# categories = [data[column].unique()
#               for column in data[categorical_columns]]
# OrdinalEncoder(categories=categories)
# ```

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

# %%
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_exclude=["int", "float"])
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]

# %%
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# TODO: write me!
