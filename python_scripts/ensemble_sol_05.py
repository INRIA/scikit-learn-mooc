# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

hist_gbdt = HistGradientBoostingRegressor(
    max_iter=1000, early_stopping=True)

# %%
from sklearn.model_selection import RandomizedSearchCV

params = {
    "max_depth": [3, 5, 8],
    "max_leaf_nodes": [15, 31, 69],
    "learning_rate": [0.1, 0.5, 1],
}

search = RandomizedSearchCV(hist_gbdt, params, n_jobs=-1)
