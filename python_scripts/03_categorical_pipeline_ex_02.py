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
# # 📝 Exercise 02
#
# The goal of this exercise is to evaluate the impact of feature preprocessing
# on a pipeline that uses a decision-tree-based classifier instead of logistic
# regression.
#
# - The first question is to empirically evaluate whether scaling numerical
#   feature is helpful or not;
# - The second question is to evaluate whether it is empirically better (both
#   from a computational and a statistical perspective) to use integer coded or
#   one-hot encoded categories.
#
# Hint: `HistGradientBoostingClassifier` does not yet support sparse input
# data. You might want to use `OneHotEncoder(categories=categories,
# sparse=False)` to force the use a dense representation as a workaround.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

# %%
target_name = "class"
target = df[target_name]
data = df.drop(columns=[target_name, "fnlwgt", "education-num"])

# %% [markdown]
# As in the previous notebooks, we use the utility `make_column_selector`
# to only select column with a specific data type. Besides, we list in
# advance all categories for the categorical columns.

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)
numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

categories = [
    data[column].unique() for column in data[categorical_columns]]

# %% [markdown]
# ## Reference pipeline (no numerical scaling and integer-coded categories)
#
# First let's time the pipeline we used in the main notebook to serve as a
# reference:

# %%
# %%time
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

categorical_preprocessor = OrdinalEncoder(categories=categories)
preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns)],
    remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())
scores = cross_val_score(model, data, target)
print(f"The different scores obtained are: \n{scores}")
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# ## Scaling numerical features
#
# Let's write a similar pipeline that also scales the numerical features using
# `StandardScaler` (or similar):

# %%
# Write your code here.

# %% [markdown]
# ## One-hot encoding of categorical variables
#
# For linear models, we have observed that integer coding of categorical
# variables can be very detrimental. However for
# `HistGradientBoostingClassifier` models, it does not seem to be the case as
# the cross-validation of the reference pipeline with `OrdinalEncoder` is good.
#
# Let's see if we can get an even better accuracy with `OneHotEncoder`.
#
# Reminder: in order to avoid creating fully correlated features it is
# preferable to use a `OneHotEncoder` using the option `drop="if_binary"`.
#
# Hint: `HistGradientBoostingClassifier` does not yet support sparse input
# data. You might want to use
# `OneHotEncoder(categories=categories, sparse=False)` to force the use a
# dense representation as a workaround.

# %%
# Write your code here.
