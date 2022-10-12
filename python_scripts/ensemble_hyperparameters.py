# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Hyperparameter tuning
#
# In the previous section, we did not discuss the parameters of random forest
# and gradient-boosting. However, there are a couple of things to keep in mind
# when setting these.
#
# This notebook gives crucial information regarding how to set the
# hyperparameters of both random forest and gradient boosting decision tree
# models.
#
# ```{caution}
# For the sake of clarity, no cross-validation will be used to estimate the
# variability of the testing error. We are only showing the effect of the
# parameters on the validation set of what should be the inner loop of a nested
# cross-validation.
# ```
#
# We will start by loading the california housing dataset.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

# %% [markdown]
# ## Random forest
#
# The main parameter to select in random forest is the `n_estimators` parameter.
# In general, the more trees in the forest, the better the generalization
# performance will be. However, it will slow down the fitting and prediction
# time. The goal is to balance computing time and generalization performance
# when setting the number of estimators. Here, we fix `n_estimators=100`, which
# is already the default value.
#
# ```{caution}
# Tuning the `n_estimators` for random forests generally result in a waste of
# computer power. We just need to ensure that it is large enough so that doubling
# its value does not lead to a significant improvement of the validation error.
# ```
#
# Instead, we can tune the hyperparameter `max_features`, which controls the
# size of the random subset of features to consider when looking for the best
# split when growing the trees: smaller values for `max_features` will lead to
# more random trees with hopefully more uncorrelated prediction errors. However
# if `max_features` is too small, predictions can be too random, even after
# averaging with the trees in the ensemble.
#
# If `max_features` is set to `None`, then this is equivalent to setting
# `max_features=n_features` which means that the only source of randomness in
# the random forest is the bagging procedure.

# %%
print(f"In this case, n_features={len(data.columns)}")

# %% [markdown]
# We can also tune the different parameters that control the depth of each tree
# in the forest. Two parameters are important for this: `max_depth` and
# `max_leaf_nodes`. They differ in the way they control the tree structure.
# Indeed, `max_depth` will enforce to have a more symmetric tree, while
# `max_leaf_nodes` does not impose such constraint. If `max_leaf_nodes=None`
# then the number of leaf nodes is unlimited.
#
# The hyperparameter `min_samples_leaf` controls the minimum number of samples
# required to be at a leaf node. This means that a split point (at any depth) is
# only done if it leaves at least `min_samples_leaf` training samples in each of
# the left and right branches. A small value for `min_samples_leaf` means that
# some samples can become isolated when a tree is deep, promoting overfitting. A
# large value would prevent deep trees, which can lead to underfitting.
#
# Be aware that with random forest, trees are expected to be deep since we are
# seeking to overfit each tree on each bootstrap sample. Overfitting is
# mitigated when combining the trees altogether, whereas assembling underfitted
# trees (i.e. shallow trees) might also lead to an underfitted forest.

# %%
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_distributions = {
    "max_features": [1, 2, 3, 5, None],
    "max_leaf_nodes": [10, 100, 1000, None],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
}
search_cv = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=2), param_distributions=param_distributions,
    scoring="neg_mean_absolute_error", n_iter=10, random_state=0, n_jobs=2,
)
search_cv.fit(data_train, target_train)

columns = [f"param_{name}" for name in param_distributions.keys()]
columns += ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = cv_results["std_test_score"]
cv_results[columns].sort_values(by="mean_test_error")

# %% [markdown]
# We can observe in our search that we are required to have a large number of
# `max_leaf_nodes` and thus deep trees. This parameter seems particularly
# impactful with respect to the other tuning parameters, but large values of
# `min_samples_leaf` seem to reduce the performance of the model.
#
# In practice, more iterations of random search would be necessary to precisely
# assert the role of each parameters. Using `n_iter=10` is good enough to
# quickly inspect the hyperparameter combinations that yield models that work
# well enough without spending too much computational resources. Feel free to
# try more interations on your own.
#
# Once the `RandomizedSearchCV` has found the best set of hyperparameters, it
# uses them to refit the model using the full training set. To estimate the
# generalization performance of the best model it suffices to call `.score` on
# the unseen data.

# %%
error = -search_cv.score(data_test, target_test)
print(f"On average, our random forest regressor makes an error of {error:.2f} k$")

# %% [markdown]
# ## Gradient-boosting decision trees
#
# For gradient-boosting, parameters are coupled, so we cannot set the parameters
# one after the other anymore. The important parameters are `n_estimators`,
# `learning_rate`, and `max_depth` or `max_leaf_nodes` (as previously discussed
# random forest).
#
# Let's first discuss the `max_depth` (or `max_leaf_nodes`) parameter. We saw
# in the section on gradient-boosting that the algorithm fits the error of the
# previous tree in the ensemble. Thus, fitting fully grown trees would be
# detrimental. Indeed, the first tree of the ensemble would perfectly fit
# (overfit) the data and thus no subsequent tree would be required, since there
# would be no residuals. Therefore, the tree used in gradient-boosting should
# have a low depth, typically between 3 to 8 levels, or few leaves ($2^3=8$ to
# $2^8=256$). Having very weak learners at each step will help reducing
# overfitting.
#
# With this consideration in mind, the deeper the trees, the faster the
# residuals will be corrected and less learners are required. Therefore,
# `n_estimators` should be increased if `max_depth` is lower.
#
# Finally, we have overlooked the impact of the `learning_rate` parameter until
# now. When fitting the residuals, we would like the tree to try to correct all
# possible errors or only a fraction of them. The learning-rate allows you to
# control this behaviour. A small learning-rate value would only correct the
# residuals of very few samples. If a large learning-rate is set (e.g., 1), we
# would fit the residuals of all samples. So, with a very low learning-rate, we
# will need more estimators to correct the overall error. However, a too large
# learning-rate tends to obtain an overfitted ensemble, similar to having a too
# large tree depth.

# %%
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingRegressor

param_distributions = {
    "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
    "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    "learning_rate": loguniform(0.01, 1),
}
search_cv = RandomizedSearchCV(
    GradientBoostingRegressor(), param_distributions=param_distributions,
    scoring="neg_mean_absolute_error", n_iter=20, random_state=0, n_jobs=2
)
search_cv.fit(data_train, target_train)

columns = [f"param_{name}" for name in param_distributions.keys()]
columns += ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = cv_results["std_test_score"]
cv_results[columns].sort_values(by="mean_test_error")

# %% [markdown]
#
# ```{caution}
# Here, we tune the `n_estimators` but be aware that is better to use
# `early_stopping` as done in the Exercise M6.04.
# ```
#
# In this search, we see that the `learning_rate` is required to be large
# enough, i.e. > 0.1. We also observe that for the best ranked models, having a
# smaller `learning_rate`, will require more trees or a larger number of
# leaves for each tree. However, it is particularly difficult to draw
# more detailed conclusions since the best value of an hyperparameter depends
# on the other hyperparameter values.

# %% [markdown]
# Now we estimate the generalization performance of the best model using the
# test set.

# %%
error = -search_cv.score(data_test, target_test)
print(f"On average, our GBDT regressor makes an error of {error:.2f} k$")

# %% [markdown]
# The mean test score in the held-out test set is slightly better than the score
# of the best model. The reason is that the final model is refitted on the whole
# training set and therefore, on more data than the cross-validated models of
# the grid search procedure.
