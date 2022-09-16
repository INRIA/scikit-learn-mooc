# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M3.02
#
# The goal is to find the best set of hyperparameters which maximize the
# generalization performance on a training set.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# In this exercise, we will progressively define the regression pipeline
# and later tune its hyperparameters.
#
# Start by defining pipeline that:
# * uses a `StandardScaler` to normalize the numerical data;
# * uses a `sklearn.neighbors.KNeighborsRegressor` as a predictive model.

# %% tags=["solution"]
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

scaler = StandardScaler()
model = make_pipeline(scaler, KNeighborsRegressor())

# %% [markdown]
# Use `RandomizedSearchCV` with `n_iter=20` to find the best set of
# hyperparameters by tuning the following parameters of the `model`:
#
# - the parameter `n_neighbors` of the `KNeighborsRegressor` with values
#   `[1, 3, 5, 7, 10, 12, 15]`
# - the parameter `with_mean` of the `StandardScaler` with possible values
#   `True` or `False`;
# - the parameter `with_std` of the `StandardScaler` with possible values
#   `True` or `False`.
#
# Notice that in the notebook "Hyperparameter tuning by randomized-search" we
# pass distributions to be sampled by the `RandomizedSearchCV`. In this case we
# define a fixed grid of hyperparameters to be explored. Using a `GridSearchCV`
# instead would explore all the possible combinations on the grid, which can be
# costly to compute for large grids, whereas the parameter `n_iter` of the
# `RandomizedSearchCV` controls the number of different random combination that
# are evaluated. Notice that setting `n_iter` larger than the number of possible
# combinations in a grid (in this case 7 x 2 x 2 = 28) would lead to repeating
# already-explored combinations.
#
# Once the computation has completed, print the best combination of parameters
# stored in the `best_params_` attribute.

# %% tags=["solution"]
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "kneighborsregressor__n_neighbors" : [1, 3, 5, 7, 10, 15, 20],
    "standardscaler__with_mean": [True, False],
    "standardscaler__with_std": [True, False],
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions,
    n_iter=20, n_jobs=2, verbose=1, random_state=1)
model_random_search.fit(data_train, target_train)
model_random_search.best_params_

# %% [markdown] tags=["solution"]
# So the best hyperparameters give a model where the features are scaled but not
# centered.
#
# Getting the best parameter combinations is the main outcome of the
# hyper-parameter optimization procedure. However it is also interesting to
# assess the sensitivity of the best models to the choice of those parameters.
# The following code, not required to answer the quiz question shows how to
# conduct such an interactive analysis for this this pipeline using a parallel
# coordinate plot using the `plotly` library.
#
# We could use `cv_results = model_random_search.cv_results_` to make a
# parallel coordinate plot as we did in the previous notebook (you are more
# than welcome to try!).

# %% tags=["solution"]
import pandas as pd

cv_results = pd.DataFrame(model_random_search.cv_results_)

# %% [markdown] tags=["solution"]
# To simplify the axis of the plot, we will rename the column of the dataframe
# and only select the mean test score and the value of the hyperparameters.

# %% tags=["solution"]
column_name_mapping = {
    "param_kneighborsregressor__n_neighbors": "n_neighbors",
    "param_standardscaler__with_mean": "centering",
    "param_standardscaler__with_std": "scaling",
    "mean_test_score": "mean test score",
}

cv_results = cv_results.rename(columns=column_name_mapping)
cv_results = cv_results[column_name_mapping.values()].sort_values(
    "mean test score", ascending=False)

# %% [markdown] tags=["solution"]
# In addition, the parallel coordinate plot from `plotly` expects all data to
# be numeric. Thus, we convert the boolean indicator informing whether or not
# the data were centered or scaled into an integer, where True is mapped to 1
# and False is mapped to 0.

# %% tags=["solution"]
import numpy as np

column_scaler = ["centering", "scaling"]
cv_results[column_scaler] = cv_results[column_scaler].astype(np.int64)
cv_results

# %% tags=["solution"]
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results,
    color="mean test score",
    dimensions=["n_neighbors", "centering", "scaling", "mean test score"],
    color_continuous_scale=px.colors.diverging.Tealrose,
)
fig.show()

# %% [markdown] tags=["solution"]
# We recall that it is possible to select a range of results by clicking and
# holding on any axis of the parallel coordinate plot. You can then slide
# (move) the range selection and cross two selections to see the intersections.
#
# Selecting the best performing models (i.e. above an accuracy of ~0.68), we
# observe the following pattern:
#
# - scaling the data is important. All the best performing models are scaling
#   the data;
# - centering the data does not have a strong impact. Both approaches, centering
#   and not centering, can lead to good models;
# - using some neighbors is fine but using too many is a problem. In particular
#   no pipeline with `n_neighbors=1` can be found among the best models. In this
#   case, scaling has a bigger impact than the hyperparameter itself!
