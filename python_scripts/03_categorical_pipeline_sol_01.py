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
# # Solution for Exercise 01
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

df = pd.read_csv("../datasets/adult-census.csv")

# %%
target_name = "class"
target = df[target_name]
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]

# %%
categories = [
    data[column].unique() for column in data[categorical_columns]]

categories

# %%
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OrdinalEncoder(categories=categories),
    LogisticRegression(max_iter=1000))
scores = cross_val_score(model, data_categorical, target)
print(f"The different scores obtained are: \n{scores}")

# %%
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# Using an arbitrary mapping from string labels to integers as done here causes the linear model to make bad assumptions on the relative ordering of  categories.
#
# This prevent the model to learning anything predictive enough and the cross-validated score is even lower that the baseline we obtained by ignoring the input data and just always predict the most frequent class:

# %%
from sklearn.dummy import DummyClassifier

scores = cross_val_score(DummyClassifier(strategy="most_frequent"),
                         data_categorical, target)
print(f"The different scores obtained are: \n{scores}")
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# By comparison, a categorical encoding that does not assume any ordering in the
# categories can lead to a significantly higher score:

# %%
from sklearn.preprocessing import OneHotEncoder

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=1000))
scores = cross_val_score(model, data_categorical, target)
print(f"The different scores obtained are: \n{scores}")
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")
