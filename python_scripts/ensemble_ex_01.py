# %% [markdown]
# # 📝 Exercise 01
#
# The aim in this notebook is to investigate if we can fine-tune a bagging
# regressor and evaluate the gain obtained.
#
# We will load the california housing dataset and split it into a training and
# a testing set.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.5
)

# %% [markdown]
# Create a `BaggingRegressor` providing a `DecisionTreeRegressor` with default
# parameter as a `base_estimator`. Train the regressor and evaluate the
# performance on the testing set.

# %%
# Write your code here.

# %% [markdown]
# Now, create a `RandomizedSearchCV` instance using the previous model and
# tune the important parameters of the bagging regressor. You can list the
# parameters using `get_params`. Find the best parameters and check if you
# are able to find a set of parameters which improve the default regressor.

# %%
# Write your code here.

# %% [markdown]
# We see that the bagging regressor provides a predictor in which fine tuning
# is not as important as in the case of fitting a single decision tree.
