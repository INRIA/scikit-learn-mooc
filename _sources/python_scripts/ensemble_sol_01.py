# %% [markdown]
# # 📃 Solution of Exercise 01
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
# statistical performance on the testing set.

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

tree = DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator=tree, n_jobs=-1)
bagging.fit(data_train, target_train)
test_score = bagging.score(data_test, target_test)
print(f"Basic R2 score of the bagging regressor:\n"
      f"{test_score:.2f}")

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
for param in bagging.get_params().keys():
    print(param)

# %%
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": randint(10, 30),
    "max_samples": [0.5, 0.8, 1.0],
    "max_features": [0.5, 0.8, 1.0],
    "base_estimator__max_depth": randint(3, 10),
}
search = RandomizedSearchCV(bagging, param_grid, n_iter=20)
_ = search.fit(data_train, target_train)

# %%
import pandas as pd

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "std_test_score", "rank_test_score"]
cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results[columns].sort_values(by="rank_test_score")
cv_results

# %%
test_score = search.score(data_test, target_test)
print(f"Basic R2 score of the bagging regressor:\n"
      f"{test_score:.2f}")

# %% [markdown]
# We see that the predictor provided by the bagging regressor does not need
# much hyperparameters tuning compared to a single decision tree. We see that
# the bagging regressor provides a predictor for which fine tuning is not as
# important as in the case of fitting a single decision tree.
