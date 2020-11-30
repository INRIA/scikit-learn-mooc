# %% [markdown]
# # 📃 Solution for Exercise 03
#
# This exercise aims at studying if AdaBoost is a model that is able to
# over-fit. We will make a grid-search and check the scores by varying the
# number of estimators.
#
# We first load the california housing dataset and split it into a training
# and a testing set.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.5
)

# %% [markdown]
# Create an `AbaBoostRegressor`. Using the function
# `sklearn.model_selection.validation_curve` to get training and test scores
# by varying the value the number of estimators. *Hint: vary the number of
# estimators between 1 and 60.*

# %%
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import validation_curve

adaboost = AdaBoostRegressor()
param_range = np.unique(np.logspace(0, 1.8, num=30).astype(int))
train_scores, test_scores = validation_curve(
    adaboost, X_train, y_train, param_name="n_estimators",
    param_range=param_range, n_jobs=-1)

# %% [markdown]
# Plot both the mean training and test scores. You can as well plot the
# standard deviation of the scores.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_scores_mean, label="Training score")
plt.plot(param_range, test_scores_mean, label="Cross-validation score")

plt.fill_between(param_range,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.3)
plt.fill_between(param_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.3)

plt.legend()
plt.ylabel("$R^2$ score")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for AdaBoost regressor")

# %% [markdown]
# Plotting the validation curve, we can see that AdaBoost is not immune against
# overfitting. Indeed, there is an optimal number of estimators to be found.
# Adding to much estimators are detrimental for the performance of the model.

# %% [markdown]
# Repeat the experiment using a random forest instead of an AdaBoost regressor.

# %%
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
train_scores, test_scores = validation_curve(
    adaboost, X_train, y_train, param_name="n_estimators",
    param_range=param_range, n_jobs=-1)

# %%
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_scores_mean, label="Training score")
plt.plot(param_range, test_scores_mean, label="Cross-validation score")

plt.fill_between(param_range,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.3)
plt.fill_between(param_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.3)

plt.legend()
plt.ylabel("$R^2$ score")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for RandomForest regressor")

# %% [markdown]
# In contrary to the AdaBoost regressor, we can see that increasing the number
# trees in the forest will increase the performance of the random forest.
# In fact, a random forest has less chance to suffer from overfitting than
# AdaBoost when increasing the number of estimators.
