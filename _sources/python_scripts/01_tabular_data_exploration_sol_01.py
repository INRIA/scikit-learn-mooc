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
# # 📃 Solution for Exercise M1.01

# %% [markdown]
# Imagine we are interested in predicting penguins species based on two of
# their body measurements: culmen length and culmen depth. First we want to do
# some data exploration to get a feel for the data.
#
# What are the features? What is the target?

# %% [markdown] tags=["solution"]
# The features are "culmen length" and "culmen depth".
# The target is the penguin species.

# %% [markdown]
# The data is located in `../datasets/penguins_classification.csv`, load it
# with `pandas` into a `DataFrame`.

# %%
# solution
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")

# %% [markdown]
# Show a few samples of the data
#
# How many features are numerical? How many features are categorical?

# %% [markdown] tags=["solution"]
# Both features, "culmen length" and "culmen depth" are numerical. There are no
# categorical features in this dataset.

# %%
# solution
penguins.head()

# %% [markdown]
# What are the different penguins species available in the dataset and how many
# samples of each species are there? Hint: select the right column and use
# the [`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) method.

# %%
# solution
penguins["Species"].value_counts()

# %% [markdown]
# Plot histograms for the numerical features

# %%
# solution
_ = penguins.hist(figsize=(8, 4))

# %% [markdown]
# Show features distribution for each class. Hint: use
# [`seaborn.pairplot`](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

# %%
# solution
import seaborn

pairplot_figure = seaborn.pairplot(penguins, hue="Species")

# %% [markdown] tags=["solution"]
# We observe that the labels on the axis are overlapping. Even if it is not
# the priority of this notebook, one can tweak the by increasing the height
# of each subfigure.

# %% tags=["solution"]
pairplot_figure = seaborn.pairplot(
    penguins, hue="Species", height=4)

# %% [markdown]
# Looking at these distributions, how hard do you think it will be to classify
# the penguins only using "culmen depth" and "culmen length"?

# %% [markdown] tags=["solution"]
# Looking at the previous scatter-plot showing "culmen length" and "culmen
# depth", the species are reasonably well separated:
# - low culmen length -> Adelie
# - low culmen depth -> Gentoo
# - high culmen depth and high culmen length -> Chinstrap
#
# There is some small overlap between the species, so we can expect a
# statistical model to perform well on this dataset but not perfectly.
