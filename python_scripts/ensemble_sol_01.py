# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M6.01
#
# The aim of this notebook is to investigate if we can tune the hyperparameters
# of a bagging regressor and evaluate the gain obtained.
#
# We will load the California housing dataset and split it into a training and
# a testing set.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# Create a `BaggingRegressor` and provide a `DecisionTreeRegressor`
# to its parameter `base_estimator`. Train the regressor and evaluate its
# generalization performance on the testing set using the mean absolute error.

# %%
# solution
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

tree = DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator=tree, n_jobs=2)
bagging.fit(data_train, target_train)
target_predicted = bagging.predict(data_test)
print(f"Basic mean absolute error of the bagging regressor:\n"
      f"{mean_absolute_error(target_test, target_predicted):.2f} k$")

# %% [markdown]
# Now, create a `RandomizedSearchCV` instance using the previous model and
# tune the important parameters of the bagging regressor. Find the best
# parameters  and check if you are able to find a set of parameters that
# improve the default regressor still using the mean absolute error as a
# metric.

# ```{tip}
# You can list the bagging regressor's parameters using the `get_params`
# method.
# ```

# %%
# solution
for param in bagging.get_params().keys():
    print(param)

# %% tags=["solution"]
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": randint(10, 30),
    "max_samples": [0.5, 0.8, 1.0],
    "max_features": [0.5, 0.8, 1.0],
    "base_estimator__max_depth": randint(3, 10),
}
search = RandomizedSearchCV(
    bagging, param_grid, n_iter=20, scoring="neg_mean_absolute_error"
)
_ = search.fit(data_train, target_train)

# %% tags=["solution"]
import pandas as pd

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(search.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = cv_results["std_test_score"]
cv_results[columns].sort_values(by="mean_test_error")

# %% tags=["solution"]
target_predicted = search.predict(data_test)
print(f"Mean absolute error after tuning of the bagging regressor:\n"
      f"{mean_absolute_error(target_test, target_predicted):.2f} k$")

# %% [markdown] tags=["solution"]
# We see that the predictor provided by the bagging regressor does not need
# much hyperparameter tuning compared to a single decision tree.
