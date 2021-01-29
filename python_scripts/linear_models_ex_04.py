# %% [markdown]
# # 📝 Exercise 04
#
# In the previous notebook, we illustrated how the regularization
# parameter of the Ridge model needs to be optimized by hand.
#
# However, this way of optimizing hyperparameters is not effective:
# we did a single split, while cross-validation would be
# much more effective.
#
# This exercise will make you implement the same search but using the class
# `GridSearchCV`.
#
# First, we will:
#
# * load the California housing dataset;
# * split the data into training and testing sets;
# * create a machine learning pipeline composed of a standard scaler to
#   normalize the data, and a ridge regression as a linear model.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ridge = make_pipeline(StandardScaler(), Ridge())

# %% [markdown]
# Now the exercise is to use a `GridSearchCV` estimator to optimize the
# parameter `alpha` of the ridge regressor. Let's redefine the
# regularization parameter candidates like we did in the previous notebook.

# %%
import numpy as np

alphas = np.logspace(-1, 2, num=30)

# %% [markdown]
# You can refer to the documentation of `GridSearchCV`
# [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
# to check how to use this estimator.
# First, train the grid-search using the previous machine learning pipeline and
# print the value of the best parameter found.

# %%
# Write your code here.

# %% [markdown]
# Once you found the best parameter `alpha`, use the grid-search estimator
# that you created to predict and estimate the $R^2$ score of this model.

# %%
# Write your code here.

# %% [markdown]
# It is also interesting to know that several regressors and classifiers
# in scikit-learn are optimized to make this parameter tuning. They usually
# finish with the term "CV" for "Cross Validation" (e.g. `RidgeCV`).
#
# They are more efficient than using `GridSearchCV` and you should use them
# instead.
#
# Repeat the previous exercise using the `RidgeVC` estimator
# instead of a grid-search.
#
# Refer to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
# of `RidgeCV` for more information on how to use it.

# %%
# Write your code here.
