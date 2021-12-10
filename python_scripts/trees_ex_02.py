# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M5.02
#
# The aim of this exercise is to find out whether a decision tree
# model is able to extrapolate.
#
# By extrapolation, we refer to values predicted by a model outside of the
# range of feature values seen during the training.
#
# We will first load the regression data.

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_regression.csv")

feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data_train, target_train = penguins[[feature_name]], penguins[target_name]

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# First, create two models, a linear regression model and a decision tree
# regression model, and fit them on the training data. Limit the depth at
# 3 levels for the decision tree.

# %%
# Write your code here.

# %% [markdown]
# Create a synthetic dataset containing all possible flipper length from
# the minimum to the maximum of the training dataset. Get the predictions of
# each model using this dataset.

# %%
# Write your code here.

# %% [markdown]
# Create a scatter plot containing the training samples and superimpose the
# predictions of both models on the top.

# %%
# Write your code here.

# %% [markdown]
# Now, we will check the extrapolation capabilities of each model. Create a
# dataset containing a broader range of values than your previous dataset,
# in other words, add values below and above the minimum and the maximum of
# the flipper length seen during training.

# %%
# Write your code here.

# %% [markdown]
# Finally, make predictions with both models on this new interval of data.
# Repeat the plotting of the previous exercise.

# %%
# Write your code here.
