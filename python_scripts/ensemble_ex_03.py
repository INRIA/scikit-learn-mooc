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
# # 📝 Exercise M6.03
#
# The aim of this exercise is to:
#
# * verifying if a random forest or a gradient-boosting decision tree overfit
#   if the number of estimators is not properly chosen;
# * use the early-stopping strategy to avoid adding unnecessary trees, to
#   get the best generalization performances.
#
# We will use the California housing dataset to conduct our experiments.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# Create a gradient boosting decision tree with `max_depth=5` and
# `learning_rate=0.5`.

# %%
# Write your code here.

# %% [markdown]
#
# Also create a random forest with fully grown trees by setting `max_depth=None`.

# %%
# Write your code here.

# %% [markdown]
#
# For both the gradient-boosting and random forest models, create a validation
# curve using the training set to assess the impact of the number of trees on
# the performance of each model. Evaluate the list of parameters `param_range =
# [1, 2, 5, 10, 20, 50, 100]` and use the mean absolute error.

# %%
# Write your code here.

# %% [markdown]
# Both gradient boosting and random forest models will always improve when
# increasing the number of trees in the ensemble. However, it will reach a
# plateau where adding new trees will just make fitting and scoring slower.
#
# To avoid adding new unnecessary tree, unlike random-forest gradient-boosting
# offers an early-stopping option. Internally, the algorithm will use an
# out-of-sample set to compute the generalization performance of the model at
# each addition of a tree. Thus, if the generalization performance is not
# improving for several iterations, it will stop adding trees.
#
# Now, create a gradient-boosting model with `n_estimators=1_000`. This number
# of trees will be too large. Change the parameter `n_iter_no_change` such
# that the gradient boosting fitting will stop after adding 5 trees that do not
# improve the overall generalization performance.

# %%
# Write your code here.

# %% [markdown]
# Estimate the generalization performance of this model again using
# the `sklearn.metrics.mean_absolute_error` metric but this time using
# the test set that we held out at the beginning of the notebook.
# Compare the resulting value with the values observed in the validation
# curve.

# %%
# Write your code here.
