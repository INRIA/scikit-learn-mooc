# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise 01
#
# The aim of this exercise is to highlight caveats to have in mind when using
# feature selection. You have to be extremely careful regarding the set of
# data on which you will compute the statistic that help you feature algorithm
# to decide which feature to select.
#
# On purpose, we will make you program the wrong way of doing feature selection
# to insights.
#
# First, you will create a completely random dataset using NumPy. Using the
# function `np.random.randn`, generate a matrix `data` containing 100 samples
# and 100,000 features. Then, using the function `np.random.randint`, generate
# a vector `target` with 100 samples containing either 0 or 1.
#
# This type of dimensionality is typical in bioinformatics when dealing with
# RNA-seq. However, we will use completely randomized features such that we
# don't have a link between the data and the target. Thus, the generalization
# performance of any machine-learning model should not perform better than the
# chance-level.

# %%
import numpy as np

# Write your code here.

# %% [markdown]
# Now, create a logistic regression model and use cross-validation to check
# the score of such model. It will allow use to confirm that our model cannot
# predict anything meaningful from random data.

# %%
# Write your code here.

# %% [markdown]
# Now, we will ask you to program the **wrong** pattern to select feature.
# Select the feature by using the entire dataset. We will choose ten features
# with the highest ANOVA F-score computed on the full dataset. Subsequently,
# subsample the dataset `data` by selecting the features' subset. Finally,
# train and test a logistic regression model.
#
# You should get some surprising results.

# %%
# Write your code here.

# %% [markdown]
# Now, we will make you program the **right** way to do the feature selection.
# First, split the dataset into a training and testing set. Then, fit the
# feature selector on the training set. Then, transform both the training and
# testing sets before to train and test the logistic regression.

# %%
# Write your code here.

# %% [markdown]
# However, the previous case is not perfect. For instance, if we were asking
# to perform cross-validation, the manual `fit`/`transform` of the datasets
# will make our life hard. Indeed, the solution here is to use a scikit-learn
# pipeline in which the feature selection will be a pre processing stage
# before to train the model.
#
# Thus, start by creating a pipeline with the feature selector and the logistic
# regression. Then, use cross-validation to get an estimate of the uncertainty
# of your model generalization performance.

# %%
# Write your code here.
