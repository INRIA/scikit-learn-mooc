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
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M1.01

# %% [markdown]
# Imagine we are interested in predicting penguins species based on two of
# their body measurements: culmen length and culmen depth. First we want to do
# some data exploration to get a feel for the data.
#
# What are the features? What is the target?

# %% [markdown]
# The data is located in `../datasets/penguins_classification.csv`, load it
# with `pandas` into a `DataFrame`.

# %%
# Write your code here.

# %% [markdown]
# Show a few samples of the data
#
# How many features are numerical? How many features are categorical?

# %%
# Write your code here.

# %% [markdown]
# What are the different penguins species available in the dataset and how many
# samples of each species are there? Hint: select the right column and use
# the [`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) method.

# %%
# Write your code here.

# %% [markdown]
# Plot histograms for the numerical features

# %%
# Write your code here.

# %% [markdown]
# Show features distribution for each class. Hint: use
# [`seaborn.pairplot`](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

# %%
# Write your code here.

# %% [markdown]
# Looking at these distributions, how hard do you think it will be to classify
# the penguins only using "culmen depth" and "culmen length"?
