# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M4.02
#
# In the previous notebook, we showed that we can add new features based on the
# original feature to make the model more expressive, for instance `x ** 2` or `x ** 3`.
# In that case we only used a single feature in `data`.
#
# The aim of this notebook is to train a linear regression algorithm on a
# dataset with more than a single feature. In such a "multi-dimensional" feature
# space we can derive new features of the form `x1 * x2`, `x2 * x3`,
# etc. Products of features are usually called "non-linear or
# multiplicative interactions" between features.
#
# Feature engineering can be an important step of a model pipeline as long as
# the new features are expected to be predictive. For instance, think of a
# classification model to decide if a patient has risk of developing a heart
# disease. This would depend on the patient's Body Mass Index which is defined
# as `weight / height ** 2`.
#
# We load the dataset penguins dataset. We first use a set of 3 numerical
# features to predict the target, i.e. the body mass of the penguin.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins.csv")

columns = ["Flipper Length (mm)", "Culmen Length (mm)", "Culmen Depth (mm)"]
target_name = "Body Mass (g)"

# Remove lines with missing values for the columns of interest
penguins_non_missing = penguins[columns + [target_name]].dropna()

data = penguins_non_missing[columns]
target = penguins_non_missing[target_name]
data.head()

# %% [markdown]
# Now it is your turn to train a linear regression model on this dataset. First,
# create a linear regression model.

# %%
# Write your code here.

# %% [markdown]
# Execute a cross-validation with 10 folds and use the mean absolute error (MAE)
# as metric.

# %%
# Write your code here.

# %% [markdown]
# Compute the mean and std of the MAE in grams (g).

# %%
# Write your code here.

# %% [markdown]
# Now create a pipeline using `make_pipeline` consisting of a
# `PolynomialFeatures` and a linear regression. Set `degree=2` and
# `interaction_only=True` to the feature engineering step. Remember not to
# include the bias to avoid redundancies with the linear's regression intercept.
#
# Use the same strategy as before to cross-validate such a pipeline.

# %%
# Write your code here.

# %% [markdown]
# Compute the mean and std of the MAE in grams (g) and compare with the results
# without feature engineering.

# %%
# Write your code here.
