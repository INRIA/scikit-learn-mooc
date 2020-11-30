# %% [markdown]
# # 📝 Exercise 04
#
# The aim of this exercise is to study:
#
# * if a GBDT tends to overfit if the number of estimators is not appropriate
#   as previously saw for AdaBoost;
# * use the early-stopping strategy which avoid for grid-searching the best
#   number of parameters.
#
# We will use california housing to conduct our experiments

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.5
)

# %% [markdown]
# Similarly to the previous exercise, create a gradient boosting decision tree
# and create a validation curve to assess the impact of the number of trees
# on the performance of the model.

# %%
# Write your code here.

# %% [markdown]
# Unlike AdaBoost, the gradient boosting model will always improve when
# increasing the number of trees in the ensemble. However, it will reach a
# plateau where adding new trees will just make fitting and scoring slower.
#
# To avoid adding new unnecessary tree, gradient boosting offers an
# early-stopping option. Internally, the algorithm will use an out-of-sample
# set to compute the performance of the model at each addition of a tree.
# Thus, if the the performance are not improving for several iterations, one
# could stop adding tree.
#
# Now, create a gradient-boosting model with `n_estimators=1000`. This number
# of trees will be too large. Change the parameter `n_iter_no_change` such
# that the gradient boosting fitting will stop after adding 5 trees that do not
# allow to improve the overall performance.

# %%
# Write your code here.
