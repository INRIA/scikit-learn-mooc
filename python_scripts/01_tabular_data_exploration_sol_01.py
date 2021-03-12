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
