# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M7.01
#
# This notebook aims at building baseline classifiers, which we'll use to
# compare our predictive model. Besides, we will check the differences with
# the baselines that we saw in regression.
#
# We will use the adult census dataset, using only the numerical features.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric-all.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]

# %% [markdown]
# First, define a `ShuffleSplit` cross-validation strategy taking half of the
# sample as a testing at each round.

# %%
# Write your code here.

# %% [markdown]
# Next, create a machine learning pipeline composed of a transformer to
# standardize the data followed by a logistic regression.

# %%
# Write your code here.

# %% [markdown]
# Get the test score by using the model, the data, and the cross-validation
# strategy that you defined above.

# %%
# Write your code here.

# %% [markdown]
# Using the `sklearn.model_selection.permutation_test_score` function,
# check the chance level of the previous model.

# %%
# Write your code here.

# %% [markdown]
# Finally, compute the test score of a dummy classifier which would predict
# the most frequent class from the training set. You can look at the
# `sklearn.dummy.DummyClassifier` class.

# %%
# Write your code here.

# %% [markdown]
# Now that we collected the results from the baselines and the model, plot
# the distributions of the different test scores.

# %% [markdown]
# We concatenate the different test score in the same pandas dataframe.

# %%
# Write your code here.

# %% [markdown]
# Next, plot the distributions of the test scores.

# %%
# Write your code here.

# %% [markdown]
# Change the strategy of the dummy classifier to `stratified`, compute the
# results and plot the distribution together with the other results. Explain
# why the results get worse.

# %%
# Write your code here.
