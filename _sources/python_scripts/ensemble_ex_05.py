# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M6.05
#
# The aim of the exercise is to get familiar with the histogram
# gradient-boosting in scikit-learn. Besides, we will use this model within
# a cross-validation framework in order to inspect internal parameters found
# via grid-search.
#
# We will use the California housing dataset.

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

# %% [markdown]
# First, create a histogram gradient boosting regressor. You can set the
# trees number to be large, and configure the model to use early-stopping.

# %%
# Write your code here.

# %% [markdown]
# We will use a grid-search to find some optimal parameter for this model.
# In this grid-search, you should search for the following parameters:
#
# * `max_depth: [3, 8]`;
# * `max_leaf_nodes: [15, 31]`;
# * `learning_rate: [0.1, 1]`.
#
# Feel free to explore the space with additional values. Create the
# grid-search providing the previous gradient boosting instance as the model.

# %%
# Write your code here.

# %% [markdown]
# Finally, we will run our experiment through cross-validation. In this regard,
# define a 5-fold cross-validation. Besides, be sure to shuffle the data.
# Subsequently, use the function `sklearn.model_selection.cross_validate`
# to run the cross-validation. You should also set `return_estimator=True`,
# so that we can investigate the inner model trained via cross-validation.

# %%
# Write your code here.

# %% [markdown]
# Now that we got the cross-validation results, print out the mean and
# standard deviation score.

# %%
# Write your code here.

# %% [markdown]
# Then inspect the `estimator` entry of the results and check the best
# parameters values. Besides, check the number of trees used by the model.

# %%
# Write your code here.

# %% [markdown]
# Inspect the results of the inner CV for each estimator of the outer CV.
# Aggregate the mean test score for each parameter combination and make a box
# plot of these scores.

# %%
# Write your code here.
