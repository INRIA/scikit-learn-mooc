# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor()

# %%
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, scoring=["r2", "neg_median_absolute_error"], cv=10, n_jobs=-1
)

# %%
