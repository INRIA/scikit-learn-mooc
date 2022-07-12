# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M6.02
#
# The aim of this exercise it to explore some attributes available in
# scikit-learn's random forest.
#
# First, we will fit the penguins regression dataset.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("../datasets/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data, target = penguins[[feature_name]], penguins[target_name]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# Create a random forest containing three trees. Train the forest and
# check the generalization performance on the testing set in terms of mean
# absolute error.

# %%
# Write your code here.

# %% [markdown]
# We now aim to plot the predictions from the individual trees in the forest.
# For that purpose you have to create first a new dataset containing evenly
# spaced values for the flipper length over the interval between 170 mm and 230
# mm.

# %%
# Write your code here.

# %% [markdown]
# The trees contained in the forest that you created can be accessed with the
# attribute `estimators_`. Use them to predict the body mass corresponding to
# the values in this newly created dataset. Similarly find the predictions of
# the random forest in this dataset.

# %%
# Write your code here.

# %% [markdown]
# Now make a plot that displays:
# - the whole `data` using a scatter plot;
# - the decision of each individual tree;
# - the decision of the random forest.

# %%
# Write your code here.
