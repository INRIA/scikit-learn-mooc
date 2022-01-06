# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M7.01
#
# In this exercise we will define dummy classification baselines and use them
# as reference to assess the relative predictive performance of a given model
# of interest.
#
# We illustrate those baselines with the help of the Adult Census dataset,
# using only the numerical features for the sake of simplicity.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric-all.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]

# %% [markdown]
# First, define a `ShuffleSplit` cross-validation strategy taking half of the
# samples as a testing at each round. Let us use 10 cross-validation rounds.

# %%
# Write your code here.

# %% [markdown]
# Next, create a machine learning pipeline composed of a transformer to
# standardize the data followed by a logistic regression classifier.

# %%
# Write your code here.

# %% [markdown]
# Compute the cross-validation (test) scores for the classifier on this
# dataset. Store the results pandas Series as we did in the previous notebook.

# %%
# Write your code here.

# %% [markdown]
# Now, compute the cross-validation scores of a dummy classifier that
# constantly predicts the most frequent class observed the training set. Please
# refer to the online documentation for the [sklearn.dummy.DummyClassifier
# ](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
# class.
#
# Store the results in a second pandas Series.

# %%
# Write your code here.

# %% [markdown]
# Now that we collected the results from the baseline and the model,
# concatenate the test scores as columns a single pandas dataframe.

# %%
# Write your code here.

# %% [markdown]
#
# Next, plot the histogram of the cross-validation test scores for both
# models with the help of [pandas built-in plotting
# function](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#histograms).
#
# What conclusions do you draw from the results?

# %%
# Write your code here.

# %% [markdown]
# Change the `strategy` of the dummy classifier to `"stratified"`, compute the
# results. Similarly compute scores for `strategy="uniform"` and then the  plot
# the distribution together with the other results.
#
# Are those new baselines better than the previous one? Why is this the case?
#
# Please refer to the scikit-learn documentation on
# [sklearn.dummy.DummyClassifier](
# https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
# to find out about the meaning of the `"stratified"` and `"uniform"`
# strategies.

# %%
# Write your code here.

# %%
# Write your code here.
