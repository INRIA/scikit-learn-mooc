# %% [markdown]
# # 📃 Solution for Exercise 03
#
# This exercise aims at verifying if AdaBoost can over-fit.
# We will make a grid-search and check the scores by varying the
# number of estimators.
#
# We will first load the California housing dataset and split it into a
# training and a testing set.

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
# Then, create an `AbaBoostRegressor`. Use the function
# `sklearn.model_selection.validation_curve` to get training and test scores
# by varying the number of estimators.
# *Hint: vary the number of estimators between 1 and 60.*

# %%
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import validation_curve

adaboost = AdaBoostRegressor()
param_range = np.unique(np.logspace(0, 1.8, num=30).astype(int))
train_scores, test_scores = validation_curve(
    adaboost, data_train, target_train, param_name="n_estimators",
    param_range=param_range, n_jobs=-1)

# %% [markdown]
# Plot both the mean training and test scores. You can also plot the
# standard deviation of the scores.

# %%
import matplotlib.pyplot as plt

plt.errorbar(param_range, train_scores.mean(axis=1),
             yerr=train_scores.std(axis=1), label="Training score",
             alpha=0.7)
plt.errorbar(param_range, test_scores.mean(axis=1),
             yerr=test_scores.std(axis=1), label="Cross-validation score",
             alpha=0.7)

plt.legend()
plt.ylabel("$R^2$ score")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for AdaBoost regressor")

# %% [markdown]
# Plotting the validation curve, we can see that AdaBoost is not immune against
# overfitting. Indeed, there is an optimal number of estimators to be found.
# Adding too many estimators is detrimental for the performance of the model.

# %% [markdown]
# Repeat the experiment using a random forest instead of an AdaBoost regressor.

# %%
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
train_scores, test_scores = validation_curve(
    forest, data_train, target_train, param_name="n_estimators",
    param_range=param_range, n_jobs=-1)

# %%
plt.errorbar(param_range, train_scores.mean(axis=1),
             yerr=train_scores.std(axis=1), label="Training score",
             alpha=0.7)
plt.errorbar(param_range, test_scores.mean(axis=1),
             yerr=test_scores.std(axis=1), label="Cross-validation score",
             alpha=0.7)

plt.legend()
plt.ylabel("$R^2$ score")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for RandomForest regressor")

# %% [markdown]
# In contrary to the AdaBoost regressor, we can see that increasing the number
# trees in the forest will increase the statistical performance of the random
# forest. In fact, a random forest has less chance to suffer from overfitting
# than AdaBoost when increasing the number of estimators.
