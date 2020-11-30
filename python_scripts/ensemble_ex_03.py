# %% [markdown]
# # 📝 Exercise 03
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
# Write your code here.

# %% [markdown]
# Plot both the mean training and test scores. You can as well plot the
# standard deviation of the scores.

# %%
# Write your code here.

# %% [markdown]
# Plotting the validation curve, we can see that AdaBoost is not immune against
# overfitting. Indeed, there is an optimal number of estimators to be found.
# Adding to much estimators are detrimental for the performance of the model.

# %% [markdown]
# Repeat the experiment using a random forest instead of an AdaBoost regressor.

# %%
# Write your code here.
