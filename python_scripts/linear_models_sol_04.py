# %% [markdown]
# # Solution for Exercise 04
#
# In the previous notebook, we illustrated how a regularization parameter of
# ridge model need to be optimized by hand. However, this way of optimizing
# hyperparameters is not effective: only a single split was used while
# cross-validation could be used.
#
# This exercise will make you implement the same search but using the class
# `GridSearchCV`.
#
# First, we will:
#
# * load the california housing dataset;
# * split the data into a training and testing set;
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
# parameter `alpha` of the ridge regressor. We redefine all the regularization
# parameter candidates that we saw in the previous notebook.

# %%
import numpy as np

alphas = np.logspace(-1, 2, num=30)

# %% [markdown]
# You can refer to the documentation of `GridSearchCV`
# [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
# to check how to use this estimator.
# First, train the grid-search using the previous machine learning pipeline and
# check what is the value of the best parameters found.

# %% [markdown]
# First, we need to check the name of the parameter to tune in the pipeline.
# We can use `get_params()` to know more about the available parameters.

# %%
for param in ridge.get_params().keys():
    print(param)

# %%
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(ridge, param_grid={"ridge__alpha": alphas})
search.fit(X_train, y_train)
print(
    f"Best alpha found via the grid-search:\n"
    f"{search.best_params_}"
)

# %% [markdown]
# Once that you found the best parameter `alpha`, use the grid-search estimator
# that you created to predict and estimate the R2 score of this model.

# %%
print(
    f"R2 score of the optimal ridge is:\n"
    f"{search.score(X_test, y_test)}"
)

# %% [markdown]
# It is also interesting to know that several regressors and classifiers
# in scikit-learn are optimized to make this parameter tuning. They usually
# finish with the term "CV" for "Cross Validation" (e.g. `RidgeCV`).
# They are more efficient than using `GridSearchCV` and you should use them
# instead.
#
# Repeat the previous exercise but using `RidgeCV` estimator instead of a
# grid-search. Refer to the documentation of `RidgeCV` to have more information
# on how to use it.

# %%
from sklearn.linear_model import RidgeCV

ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
ridge.fit(X_train, y_train)
print(
    f"Best alpha found via the grid-search:\n"
    f"{ridge[-1].alpha_}"
)

# %%
print(
    f"R2 score of the optimal ridge is:\n"
    f"{ridge.score(X_test, y_test)}"
)

# %%
