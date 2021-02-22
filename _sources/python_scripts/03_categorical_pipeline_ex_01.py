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
# The goal of this exercise is to evaluate the impact of using an arbitrary
# integer encoding for categorical variables along with a linear
# classification model such as Logistic Regression.
#
# To do so, let's try to use `OrdinalEncoder` to preprocess the categorical
# variables. This preprocessor is assembled in a pipeline with
# `LogisticRegression`. The statistical performance of the pipeline can be
# evaluated as usual by cross-validation and then compared to the score
# obtained when using `OneHotEncoder` or to some other baseline score.
#
# Because `OrdinalEncoder` can raise errors if it sees an unknown category at
# prediction time, you can set the `handle_unknown` and `unknown_value`
# parameters.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %%
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "fnlwgt", "education-num"])

# %% [markdown]
# We can select the categorical based on the `object` dtype.

# %%
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]

# %%
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# Write your code here.
