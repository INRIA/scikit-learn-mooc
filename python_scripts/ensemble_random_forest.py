# %% [markdown]
# # Random forest
#
# In this notebook, we will present random forest models and show the
# differences with a bagging classifiers.
#
# A random forest, a popular model in machine learning, is a modification of
# the bagging algorithm. In bagging, any classifier or regressor can be used.
# In a random forest, the base classifier or regressor must be a decision tree.
# In our previous example, we used a decision tree but we could have used a
# linear model as the regressor for our bagging algorithm.
#
# In addition, random forest is different from bagging when used with
# classifiers: when searching for the best split, only a subset of the original
# features are used. By default, this subset of feature is equal to the square
# root of the total number of features. In regression, the total number of
# available features will be used.
#
# We will illustrate the usage of a random forest and compare it with the
# bagging regressor on the "California housing" dataset.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.5)

# %%
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(
    n_estimators=100, random_state=0, n_jobs=-1
)
bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(random_state=0),
    n_estimators=100,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)
bagging.fit(X_train, y_train)

print(f"Performance of random forest: "
      f"{random_forest.score(X_test, y_test):.3f}")
print(f"Performance of bagging: "
      f"{bagging.score(X_test, y_test):.3f}")

# %% [markdown]
# Notice that we don't need to provide a `base_estimator` parameter to
# `RandomForestRegressor`, it is always a tree classifier. Also note that the
# scores are almost identical. This is because our problem is a regression
# problem and therefore, the number of features used in random forest and
# bagging is the same.
#
# For classification problems, we would need to pass a tree model instance
# with the parameter `max_features="sqrt"` to `BaggingRegressor` if we wanted
# it to have the same behaviour as the random forest classifier.
#
# ## Classifiers details
#
# Up to now, we have only focused on regression problems. There is a little
# difference between regression and classification.
#
# First, the `base_estimator` should be chosen in line with the problem that
# is solved: use a classifier with a classification problem and a regressor
# with a regression problem.
#
# Then, the aggregation method is different in regression and classification:
#
# - in regression, the average prediction is computed. For instance, if
#   three learners predict 0.4, 0.3 and 0.31, the aggregation will output 0.33;
# - while in classification, the class which highest probability (after
#   averaging the predicted probabilities) is predicted. For instance, if three
#   learners predict (for two classes) the probability (0.4, 0.6), (0.3, 0.7)
#   and (0.31, 0.69), the aggregation probability is (0.33, 0.67) and the
#   second class would be predicted.
#
# # Midpoint summary
#
# We saw in this section two algorithms that use bootstrap samples to create
# an ensemble of classifiers or regressors. These algorithms train several
# learners on different bootstrap samples. The predictions are then
# aggregated. This operation can be done in a very efficient manner since the
# training of each learner can be done in parallel.
