# %% [markdown]
# # 📃 Solution of Exercise 01
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

tree = DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator=tree, n_jobs=-1)
bagging.fit(X_train, y_train)

print(
    f"Basic R2 score og a bagging regressor:\n"
    f"{bagging.score(X_test, y_test):.2f}"
)

# %% [markdown]
# Now, create a `RandomizedSearchCV` instance using the previous model and
# tune the important parameters of the bagging regressor. You can list the
# parameters using `get_params`. Find the best parameters and check if you
# are able to find a set of parameters which improve the default regressor.

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
search.fit(X_train, y_train)

# %%
import pandas as pd

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "std_test_score", "rank_test_score"]
cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results[columns].sort_values(by="rank_test_score")
cv_results

# %%
print(
    f"Basic R2 score og a bagging regressor:\n"
    f"{search.score(X_test, y_test):.2f}"
)

# %% [markdown]
# We see that the bagging regressor provides a predictor in which fine tuning
# is not as important as in the case of fitting a single decision tree.
