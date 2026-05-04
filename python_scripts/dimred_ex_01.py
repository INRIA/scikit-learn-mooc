# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Solution for Exercise M8.01
#
# In the Chapter on Linear Models we saw that feature engineering using
# `PolynomialFeatures` can give a linear model the flexibility to capture
# non-linear relationships, in particular, it is useful to model interactions
# between features.
#
# The downside is that the feature space grows quadratically with `n_features`,
# and if the input features are correlated, those new features would be
# correlated as well. Fitting a linear regressor on that many correlated
# features can be unnecessarily slow and prone to overfitting, even with
# regularization.
#
# PCA after a polynomial expansion compresses that feature space into a smaller
# set of components that retain the flexibility from the non-linearly augmented
# feature space, before passing them to the final estimator.
#
# In this exercise we explore whether we can tune `n_components` when it is used
# as a preprocessing step in a supervised pipeline. For such purpose we use the
# Ames Housing dataset, keeping only a subset of the numerical features.

# %%
import pandas as pd
import numpy as np

ames_housing = pd.read_csv("../datasets/ames_housing_no_missing.csv")
numerical_features = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "MiscVal",
]
target_name = "SalePrice"
data, target = (
    ames_housing[numerical_features],
    ames_housing[target_name],
)
target /= 1000

# %% [markdown]
# First, fit a `PolynomialFeatures` with `degree=2` and `include_bias=False` to
# the whole data to check how many features are produced by the polynomial
# expansion. You can use the attributes `n_features_in_` and
# `n_output_features_`.

# %%
# Write your code here.

# %% [markdown]
# Build a pipeline with `skrub.SquashingScaler` using `quantile_range=(5.0,
# 95.0)`, followed by the previous polynomial expansion, then `PCA`, and `Ridge`
# regression with default parameters as the final predictor. Use `GridSearchCV`
# with `scoring` set to "neg_root_mean_squared_error" to search over
# `n_components` using the grid defined below. Fit it on the full dataset.

# %%
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from skrub import SquashingScaler

param_name = "pca__n_components"
param_grid = {param_name: [3, 10, 50, 100, 200, None]}

# %%
# Write your code here.

# %% [markdown]
# Build a `grid_search_results` DataFrame from the attribute `cv_results_`
# keeping only the columns listed in `results_columns` below. Add a
# `"mean_test_error"` column as the negative of `"mean_test_score"`, drop
# `"mean_test_score"`, and rename `"param_pca__n_components"` to
# `"n_components"`.

# %%
results_columns = [
    "mean_test_score",
    "std_test_score",
    "mean_fit_time",
    "std_fit_time",
    "mean_score_time",
    "std_score_time",
    "param_" + param_name,
]

# %%
# Write your code here.

grid_search_results = [...]  # complete this code

# %% [markdown]
# The following cell plots test RMSE against fit time using `grid_search_results`.
# Hover over a point to see the corresponding `n_components` value. It runs
# without modifications once `grid_search_results` is correctly defined.

# %%
import plotly.express as px

labels = {
    "mean_fit_time": "CV fit time (s)",
    "mean_test_error": "CV score (MAE)",
}
grid_search_results["n_components"] = grid_search_results[
    "n_components"
].fillna("None")
fig = px.scatter(
    grid_search_results,
    x="mean_fit_time",
    y="mean_test_error",
    error_x="std_fit_time",
    error_y="std_test_score",
    hover_data=grid_search_results.columns,
    labels=labels,
)
fig.update_layout(
    title={
        "text": "Trade-off between fit time and mean test score",
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)
fig.show(renderer="notebook")

# %% [markdown]
# ## Does reducing dimensions stabilize the optimal `alpha`?
#
# During the first part of this exercise we fixed `alpha` to focus on the
# fit-time tradeoff. As we saw in the chapter on linear models, the optimal
# regularization strength is not necessarily the same on all cross-validation
# iterations. Here we ask whether reducing the number of components makes the
# choice of alpha more stable across folds.
#
# Replace the `Ridge` from your previous pipeline to use
# `RidgeCV(alphas=alphas)`. This replaces the grid search over `alpha` with a
# faster internal selection, so we only need to loop over `n_components=[50,
# 100, None]` by hand. For each component, `cross_validate`  using
# `ShuffleSplit` (as below) and set `return_estimator` in the `cross_validate`
# function to collect the `alpha_` attribute at each split and store them in
# `best_alphas`.
#
# Does reducing dimensions make the optimal `alpha` more consistent across
# folds?

# %%
from collections import defaultdict

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

alphas = np.logspace(-2, 2, 21)
best_alphas = defaultdict(list)
cv = ShuffleSplit(n_splits=50, random_state=0)

# Write your code here.

# %% [markdown]
# Make a boxplot of `best_alphas` using a "log" `yscale`.

# %%
# Write your code here.

# %% [markdown]
# Instead of changing the `random_state`, feel free to increase `n_splits` to
# see whether the shape of the boxes changes. Just keep in mind that increases
# the running time.
