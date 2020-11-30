# %% [markdown]
# # 📝 Exercise 05
#
# The aim of the exercise is to get familiar with the histogram
# gradient-boosting in scikit-learn. Besides, we will use this model within
# a cross-validation framework in order to inspect internal parameters found
# via grid-search.
#
# We will use the california housing dataset.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# %% [markdown]
# First, create a histogram gradient boosting regressor. You can set the number
# of trees to be large enough. Indeed, you fix the parameter such that model
# will use early-stopping.

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
# Feel free to explore more the space with additional values. Create the
# grid-search providing the previous gradient boosting instance as model.

# %%
# Write your code here.

# %% [markdown]
# Finally, we will run our experiment through cross-validation. In this regard,
# define a 5-fold cross-validation. Besides, be sure to shuffle the the data.
# Subsequently, use the function `sklearn.model_selection.cross_validate`
# to run the cross-validation. You should as well set `return_estimator=True`,
# such that we can investigate the inner model trained via cross-validation.

# %%
# Write your code here.

# %% [markdown]
# We got the results of the cross-validation. First check what is the mean and
# standard deviation score.

# %%
# Write your code here.

# %% [markdown]
# Inspect the results of the inner CV for each estimator of the outer CV.
# Aggregate the mean test score for each parameter combination and make a box
# plot of these scores.

# %%
# Write your code here.
