# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 03
#
# The goal of this exercise is to evaluate the impact of feature preprocessing on a pipeline that uses a  decision-tree-based classifier instead of logistic regression.
#
# - The first question is to empirically evaluate whether scaling numerical feature is helpful or not;
#
# - The second question is to evaluate whether it is empirically better (both from a computational and a statistical perspective) to use integer coded or one-hot encoded categories.

# %%
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

# %%
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
numerical_columns = [
    c for c in data.columns if data[c].dtype.kind in ["i", "f"]]
categorical_columns = [
    c for c in data.columns if data[c].dtype.kind not in ["i", "f"]]

categories = [
    data[column].unique() for column in data[categorical_columns]]

# %% [markdown]
# ## Reference pipeline (no numerical scaling and integer-coded categories)
#
# First let's time the pipeline we used in the main notebook to serve as a reference:

# %%
# %%time

preprocessor = ColumnTransformer([
    ('categorical', OrdinalEncoder(categories=categories),
     categorical_columns),], remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())
scores = cross_val_score(model, data, target)
print(f"The different scores obtained are: \n{scores}")
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# ## Scaling numerical features
#
# Let's write a similar pipeline that also scales the numerical features using `StandardScaler` (or similar):

# %%
# TODO write me!

# %% [markdown]
# ## One-hot encoding of categorical variables
#
# For linear models, we have observed that integer coding of categorical
# variables can be very detrimental. However for
# `HistGradientBoostingClassifier` models, it does not seem to be the
# case as the cross-validation of the reference pipeline with
# `OrdinalEncoder` is good.
#
# Let's see if we can get an even better accuracy with `OneHotEncoding`.
#
# Hint: `HistGradientBoostingClassifier` does not yet support sparse input data. You might want to use
# `OneHotEncoder(handle_unknown="ignore", sparse=False)` to force the use a dense representation as a workaround.

# %%
# TODO: write me!
