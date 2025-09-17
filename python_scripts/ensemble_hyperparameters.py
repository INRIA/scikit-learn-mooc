# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Hyperparameter tuning
#
# In the previous section, we did not discuss the hyperparameters of random
# forest and histogram gradient-boosting. This notebook gives crucial
# information regarding how to set them.
#
# ```{caution}
# For the sake of clarity, no nested cross-validation is used to estimate the
# variability of the testing error. We are only showing the effect of the
# parameters on the validation set.
# ```
#
# We start by loading the california housing dataset.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0
)

# %% [markdown]
# ## Random forest
#
# The main parameter to select in random forest is the `n_estimators` parameter.
# In general, the more trees in the forest, the better the generalization
# performance would be. However, adding trees slows down the fitting and prediction
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
# split when growing the trees: smaller values for `max_features` lead to
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
# Indeed, `max_depth` enforces growing symmetric trees, while `max_leaf_nodes`
# does not impose such constraint. If `max_leaf_nodes=None` then the number of
# leaf nodes is unlimited.
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
    RandomForestRegressor(n_jobs=2),
    param_distributions=param_distributions,
    scoring="neg_mean_absolute_error",
    n_iter=10,
    random_state=0,
    # n_jobs=2,  # Uncomment this line if you run locally
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
print(
    f"On average, our random forest regressor makes an error of {error:.2f} k$"
)

# %% [markdown]
# ## Histogram gradient-boosting decision trees
#
# For gradient-boosting, hyperparameters are coupled, so we cannot set them
# one after the other anymore. The important hyperparameters are `max_iter`,
# `learning_rate`, and `max_depth` or `max_leaf_nodes` (as previously discussed
# random forest).
#
# Let's first discuss `max_iter` which, similarly to the `n_estimators`
# hyperparameter in random forests, controls the number of trees in the
# estimator. The difference is that the actual number of trees trained by the
# model is not entirely set by the user, but depends also on the stopping
# criteria: the number of trees can be lower than `max_iter` if adding a new
# tree does not improve the model enough. We will give more details on this in
# the next exercise.
#
# The depth of the trees is controlled by `max_depth` (or `max_leaf_nodes`). We
# saw in the section on gradient-boosting that boosting algorithms fit the error
# of the previous tree in the ensemble. Thus, fitting fully grown trees would be
# detrimental. Indeed, the first tree of the ensemble would perfectly fit
# (overfit) the data and thus no subsequent tree would be required, since there
# would be no residuals. Therefore, the tree used in gradient-boosting should
# have a low depth, typically between 3 to 8 levels, or few leaves ($2^3=8$ to
# $2^8=256$). Having very weak learners at each step helps reducing overfitting.
#
# With this consideration in mind, the deeper the trees, the faster the
# residuals are corrected and then less learners are required. Therefore,
# it can be beneficial to increase `max_iter` if `max_depth` is low.
#
# Finally, we have overlooked the impact of the `learning_rate` parameter
# until now. This parameter controls how much each correction contributes to the
# final prediction. A smaller learning-rate means the corrections of a new
# tree result in small adjustments to the model prediction. When the
# learning-rate is small, the model generally needs more trees to achieve good
# performance. A higher learning-rate makes larger adjustments with each tree,
# which requires fewer trees and trains faster, at the risk of overfitting. The
# learning-rate needs to be tuned by hyperparameter tuning to obtain the best
# value that results in a model with good generalization performance.

# %%
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingRegressor

param_distributions = {
    "max_iter": [3, 10, 30, 100, 300, 1000],
    "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    "learning_rate": loguniform(0.01, 1),
}
search_cv = RandomizedSearchCV(
    HistGradientBoostingRegressor(),
    param_distributions=param_distributions,
    scoring="neg_mean_absolute_error",
    n_iter=20,
    random_state=0,
    # n_jobs=2, # Uncomment this line if you run locally
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
# Here, we tune `max_iter` but be aware that it is better to set `max_iter` to a
# fixed, large enough value and use parameters linked to `early_stopping` as we
# will do in Exercise M6.04.
# ```
#
# In this search, we observe that for the best ranked models, having a
# smaller `learning_rate`, requires more trees or a larger number of leaves
# for each tree. However, it is particularly difficult to draw more detailed
# conclusions since the best value of each hyperparameter depends on the other
# hyperparameter values.

# %% [markdown]
# We can now estimate the generalization performance of the best model using the
# test set.

# %%
error = -search_cv.score(data_test, target_test)
print(f"On average, our HGBT regressor makes an error of {error:.2f} k$")

# %% [markdown]
# The mean test score in the held-out test set is slightly better than the score
# of the best model. The reason is that the final model is refitted on the whole
# training set and therefore, on more data than the cross-validated models of
# the grid search procedure.
#
# We summarize these details in the following table:
#
# | **Bagging & Random Forests**                     | **Boosting**                                        |
# |--------------------------------------------------|-----------------------------------------------------|
# | fit trees **independently**                      | fit trees **sequentially**                          |
# | each **deep tree overfits**                      | each **shallow tree underfits**                     |
# | averaging the tree predictions **reduces overfitting** | sequentially adding trees **reduces underfitting** |
# | generalization improves with the number of trees | too many trees may cause overfitting                |
# | does not have a `learning_rate` parameter        | fitting the residuals is controlled by the `learning_rate` |
