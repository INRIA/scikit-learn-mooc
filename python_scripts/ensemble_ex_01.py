# %% [markdown]
# # 📝 Exercise 01
#
# The aim of this notebook is to investigate if we can fine-tune a bagging
# regressor and evaluate the gain obtained.
#
# We will load the California housing dataset and split it into a training and
# a testing set.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)

# %% [markdown]
# Create a `BaggingRegressor` and provide a `DecisionTreeRegressor`
# to its parameter `base_estimator`. Train the regressor and evaluate its
# statistical performance on the testing set.

# %%
# Write your code here.

# %% [markdown]
# Now, create a `RandomizedSearchCV` instance using the previous model and
# tune the important parameters of the bagging regressor. Find the best
# parameters  and check if you are able to find a set of parameters that
# improve the default regressor.

# ```{tip}
# You can list the bagging regressor's parameters using the `get_params`
# method.
# ```

# %%
# Write your code here.

# %% [markdown]
# We see that the bagging regressor provides a predictor in which fine tuning
# is not as important as in the case of fitting a single decision tree.
