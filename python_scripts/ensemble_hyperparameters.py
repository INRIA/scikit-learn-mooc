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
# testing error. We are only showing the effect of the parameters
# on the validation set of what should be the inner cross-validation.
# ```
#
# ## Random forest
#
# The main parameter to tune for random forest is the `n_estimators` parameter.
# In general, the more trees in the forest, the better the generalization
# performance will be. However, it will slow down the fitting and prediction
# time. The goal is to balance computing time and generalization performance when
# setting the number of estimators when putting such learner in production.
#
# Then, we could also tune a parameter that controls the depth of each tree in
# the forest. Two parameters are important for this: `max_depth` and
# `max_leaf_nodes`. They differ in the way they control the tree structure.
# Indeed, `max_depth` will enforce to have a more symmetric tree, while
# `max_leaf_nodes` does not impose such constraint.
#
# Be aware that with random forest, trees are generally deep since we are
# seeking to overfit each tree on each bootstrap sample because this will be
# mitigated by combining them altogether. Assembling underfitted trees (i.e.
# shallow trees) might also lead to an underfitted forest.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

# %%
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_distributions = {
    "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
    "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
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
# We can observe in our search that we are required to have a large
# number of leaves and thus deep trees. This parameter seems particularly
# impactful in comparison to the number of trees for this particular dataset:
# with at least 50 trees, the generalization performance will be driven by the
# number of leaves.
#
# Now we will estimate the generalization performance of the best model by
# refitting it with the full training set and using the test set for scoring on
# unseen data. This is done by default when calling the `.fit` method.

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
# Here, we tune the `n_estimators` but be aware that using early-stopping as
# in the previous exercise will be better.
# ```
#
# In this search, we see that the `learning_rate` is required to be large
# enough, i.e. > 0.1. We also observe that for the best ranked models, having a
# smaller `learning_rate`, will require more trees or a larger number of
# leaves for each tree. However, it is particularly difficult to draw
# more detailed conclusions since the best value of an hyperparameter depends
# on the other hyperparameter values.

# %% [markdown]
# Now we estimate the generalization performance of the best model
# using the test set.

# %%
error = -search_cv.score(data_test, target_test)
print(f"On average, our GBDT regressor makes an error of {error:.2f} k$")

# %% [markdown]
# The mean test score in the held-out test set is slightly better than the score
# of the best model. The reason is that the final model is refitted on the whole
# training set and therefore, on more data than the inner cross-validated models
# of the grid search procedure.
