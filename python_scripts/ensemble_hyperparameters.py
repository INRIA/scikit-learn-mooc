# %% [markdown]
# # Hyper-parameters tuning
#
# In the previous section, we did not discuss the parameters of random forest
# and gradient-boosting. However, there are a couple of things to keep in mind
# when setting these parameters.
#
# This notebook gives crucial information regarding how to set the
# hyperparameters of both random forest and gradient boostin decision tree
# models.
#
# ## Random forest
#
# The main parameter to tune with random forest is the `n_estimators`
# parameter. In general, the more trees in the forest, the better the
# performance will be. However, it will slow down the fitting and prediction
# time. So one has to balance compute time and performance when setting the
# number of estimators when putting such learner in production.
#
# The `max_depth` parameter could also be tuned. Sometimes, there is no need to
# have fully grown trees. However, be aware that with random forest, trees are
# generally deep since we are seeking to overfit the learners on the bootstrap
# samples because this will be mitigated by combining them. Assembling
# underfitted trees (i.e. shallow trees) might also lead to an underfitted
# forest.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [10, 20, 30],
    "max_depth": [3, 5, None],
}
grid_search = GridSearchCV(
    RandomForestRegressor(n_jobs=-1), param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "rank_test_score"]
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[columns].sort_values(by="rank_test_score")

# %% [markdown]
# We can observe that in our grid-search, the largest `max_depth` together with
# largest `n_estimators` led to the best performance.
#
# ## Gradient-boosting decision tree
#
# For gradient-boosting, parameters are coupled, so we can not anymore set the
# parameters one after the other. The important parameters are `n_estimators`,
# `max_depth`, and `learning_rate`.
#
# Let's first discuss the `max_depth` parameter. We saw in the section on
# gradient-boosting that the algorithm fits the error of the previous tree in
# the ensemble. Thus, fitting fully grown trees will be detrimental. Indeed,
# the first tree of the ensemble would perfectly fit (overfit) the data and
# thus no subsequent tree would be required, since there would be no residuals.
# Therefore, the tree used in gradient-boosting should have a low depth,
# typically between 3 to 8 levels. Having very weak learners at each step will
# help reducing overfitting.
#
# With this consideration in mind, the deeper the trees, the faster the
# residuals will be corrected and less learners are required. So `n_estimators`
# should be increased if `max_depth` is lower.
#
# Finally, we have overlooked the impact of the `learning_rate` parameter up
# till now. When fitting the residuals one could choose if the tree should try
# to correct all possible errors or only a fraction of them. The learning-rate
# allows you to control this behaviour. A small learning-rate value would only
# correct the residuals of very few samples. If a large learning-rate is set
# (e.g., 1), we would fit the residuals of all samples. So, with a very low
# learning-rate, we will need more estimators to correct the overall error.
# However, a too large learning-rate tends to obtain an overfitted ensemble,
# similar to having a too large tree depth.

# %%
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    "n_estimators": [10, 30, 50],
    "max_depth": [3, 5, None],
    "learning_rate": [0.1, 1],
}
grid_search = GridSearchCV(
    GradientBoostingRegressor(), param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "rank_test_score"]
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[columns].sort_values(by="rank_test_score")

# %% [markdown]
# Here, we tune the `n_estimators` but be aware that using early-stopping as
# in the previous exercise will be better.
