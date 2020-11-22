# %% [markdown]
# # Introductory example to ensemble models
#
# This first notebook aims at emphasizing the benefit of ensemble methods over
# simple models (e.g. decision tree, linear model, etc.). Combining simple
# models result in more powerful and robust models with less hassle.
#
# We will start by loading the california housing dataset. We recall that the
# goal in this dataset is to predict the median house value in some district
# in California based on demographic and geographic data.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(as_frame=True, return_X_y=True)

# %% [markdown]
# Then, we will divide the dataset into a traning and a testing set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.5
)

# %% [markdown]
# We will train a decision tree regressor and check its performance.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
print(f"R2 score of the default tree:\n"
      f"{tree.score(X_test, y_test):.3f}")

# %% [markdown]
# We obtain fair results. However, as we previously presented in the "tree in
# depth" notebook, this model needs to be tuned to overcome over- or
# under-fitting. Indeed, the default parameters will not necessarily lead to an
# optimal decision tree. Instead of using the default value, we should search
# via cross-validation the optimal value of the important parameters such as
# `max_depth`, `min_samples_split`, or `min_samples_leaf`.
#
# We recall that we need to tune these parameters, as decision trees tend to
# overfit the training data if we grow deep trees, but there are no rules on
# what each parameter should be set to. Thus, not making a search could lead us
# to have an underfitted or overfitted model.
#
# Now, we make a grid-search to fine-tune the parameters that we mentioned
# earlier.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 10, 30, 50],
    "min_samples_leaf": [0.01, 0.05, 0.1, 1],
}
cv = 3
tree = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
)

tree.fit(X_train, y_train)

# %% [markdown]
# We can create a dataframe storing the important information collected during
# the tuning of the parameters and investigate the results.

# %%
import pandas as pd

cv_results = pd.DataFrame(tree.cv_results_)
interesting_columns = [
    "param_max_depth",
    "param_min_samples_split",
    "param_min_samples_leaf",
    "mean_test_score",
    "rank_test_score",
    "mean_fit_time",
]
cv_results = cv_results[interesting_columns].sort_values(by="rank_test_score")
cv_results

# %% [markdown]
# From theses results, we can see that the best parameters is the combination
# where the depth of the tree is not limited and the minimum number of samples
# to create a leaf is also equal to 1 (the default values) and the
# minimum number of samples to make a split of 50 (much higher than the default
# value.
#
# It is interesting to look at the total amount of time it took to fit all
# these different models. In addition, we can check the performance of the
# optimal decision tree on the left-out testing data.

# %%
total_fitting_time = (cv_results["mean_fit_time"] * cv).sum()
print(
    f"Required training time of the GridSearchCV: "
    f"{total_fitting_time:.2f} seconds"
)
print(
    f"Best R2 score of a single tree: {tree.best_score_:.3f}"
)

# %% [markdown]
# Hence, we have a model that has an $R^2$ score below 0.7. So this model is
# better than the previous default decision tree.
#
# However, the amount of time to find the best learner has an heavy
# computational cost. Indeed, it depends on the number of folds used during the
# cross-validation in the grid-search multiplied by the number of parameters.
#
# Now we will use an ensemble method called bagging. More details about this
# method will be discussed in the next section. In short, this method will use
# a base regressor (i.e. decision tree regressors) and will train several of
# them on a slightly modified version of the training set. Then, the
# predictions of all these base regressors will be combined by averaging.
#
# Here, we will use 50 decision trees and check the fitting time as well as
# the performance on the left-out testing data. It is important to note that
# we are not going to tune any parameter of the decision tree.

# %%
from time import time
from sklearn.ensemble import BaggingRegressor

base_estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
    base_estimator=base_estimator, n_estimators=50, random_state=0)

start_fitting_time = time()
bagging_regressor.fit(X_train, y_train)
elapsed_fitting_time = time() - start_fitting_time

print(f"Elapsed fitting time: {elapsed_fitting_time:.2f} seconds")
print(f"R2 score: {bagging_regressor.score(X_test, y_test):.3f}")

# %% [markdown]
# We can see that the computation time is much shorter for training the full
# ensemble than for the parameter search of a single tree. In addition, the
# score is significantly improved with a $R^2$ close to 0.8. Furthermore, note
# that this result is obtained before any parameter tuning. This shows the
# motivation behind the use of an ensemble learner: it gives a relatively good
# baseline with decent performance without any parameter tuning.
#
# Now, we will discuss in detail two ensemble families: bagging and
# boosting:
#
# * ensemble using bootstrap (e.g. bagging and random-forest);
# * ensemble using boosting (e.g. adaptive boosting and gradient-boosting
#   decision tree).
