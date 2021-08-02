# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M1.05
#
# The goal of this exercise is to evaluate the impact of feature preprocessing
# on a pipeline that uses a decision-tree-based classifier instead of a logistic
# regression.
#
# - The first question is to empirically evaluate whether scaling numerical
#   features is helpful or not;
# - The second question is to evaluate whether it is empirically better (both
#   from a computational and a statistical perspective) to use integer coded or
#   one-hot encoded categories.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %%
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

# %% [markdown]
# As in the previous notebooks, we use the utility `make_column_selector`
# to select only columns with a specific data type. Besides, we list in
# advance all categories for the categorical columns.

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)
numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

# %% [markdown]
# ## Reference pipeline (no numerical scaling and integer-coded categories)
#
# First let's time the pipeline we used in the main notebook to serve as a
# reference:

# %%
import time

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns)],
    remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

start = time.time()
cv_results = cross_validate(model, data, target)
elapsed_time = time.time() - start

scores = cv_results["test_score"]

print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f} "
      f"with a fitting time of {elapsed_time:.3f}")

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
# We observed that integer coding of categorical variables can be very
# detrimental for linear models. However, it does not seem to be the case for
# `HistGradientBoostingClassifier` models, as the cross-validation score
# of the reference pipeline with `OrdinalEncoder` is reasonably good.
#
# Let's see if we can get an even better accuracy with `OneHotEncoder`.
#
# Hint: `HistGradientBoostingClassifier` does not yet support sparse input
# data. You might want to use
# `OneHotEncoder(handle_unknown="ignore", sparse=False)` to force the use of a
# dense representation as a workaround.

# %%
# Write your code here.
