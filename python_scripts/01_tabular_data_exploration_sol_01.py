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
# # 📃 Solution for Exercise 01

# %% [markdown]
# We will try to predict penguins species based on two of their body
# measurements: culmen length and culmen depth.
#
# What are the features? What is the target?

# %% markdown
# The features are "culmen length" and "culmen depth".
# The target is the penguin species.

# %% [markdown]
# The data is located in `../datasets/penguins_classification.csv`, load it
# with `pandas` into a `DataFrame`.

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")

# %% [markdown]
# Show a few samples of the data
#
# How many features are numerical? How many features are categorical?

# %% [markdown]
# Both features, "culmen length" and "culmen depth" are numerical. There are no
# no categorical features in this dataset.

# %%
penguins.head()

# %% [markdown]
# How many samples of each species do you have?

# %%
penguins["Species"].value_counts()

# %% [markdown]
# Plot histograms for the numerical features

# %%
penguins.hist()

# %% [markdown]
# Show features distribution for each class. Looking at this distributions, how
# hard do you think it will be to classify the penguins only using "culmen
# depth" and "culmen length"?

# %%
import seaborn

seaborn.pairplot(penguins, hue="Species")
# %%

# %% [markdown]
# Looking at the scatter-plot showing both "culmen length" and "culmen depth",
# the species are reasonably well separated:
# - low culmen length -> Adelie
# - low culmen depth -> Gentoo
# - high culmen depth and high culmen length -> Chinstrap
#
# There is some small overlap between the species, so we can expect a
# statistical model to perform well on this dataset but not perfectly.
