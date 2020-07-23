# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
X, y = california_housing.data, california_housing.target

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0,
)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 10, 30, 50],
    "min_samples_leaf": [0.01, 0.05, 0.1, 1],
}
cv = 3
tree = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
)

tree.fit(X_train, y_train)

# %%
import pandas as pd

cv_results = pd.DataFrame(tree.cv_results_)
interesting_columns = [
    "param_max_depth",
    "param_min_samples_split",
    "param_min_samples_leaf",
    "mean_test_score",
    "rank_test_score",
    "mean_fit_time",
]
cv_results = cv_results[interesting_columns].sort_values(by="rank_test_score")
cv_results

# %%
total_fitting_time = (cv_results["mean_fit_time"] * cv).sum()
print(
    f"Required training time of the GridSearchCV: "
    f"{total_fitting_time:.2f} seconds"
)
print(
    f"Best R2 score of a single tree: {tree.best_score_:.3f}"
)

# %%
from time import time
from sklearn.ensemble import BaggingRegressor

base_estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
    base_estimator=base_estimator, n_estimators=50, random_state=0,
)

start_fitting_time = time()
bagging_regressor.fit(X_train, y_train)
elapsed_fitting_time = time() - start_fitting_time

print(f"Elapsed fitting time: {elapsed_fitting_time:.2f} seconds")
print(f"R2 score: {bagging_regressor.score(X_test, y_test):.3f}")

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

n_sample = 50
x_max, x_min = 1.4, -1.4
len_x = (x_max - x_min)
x = rng.rand(n_sample) * len_x - len_x/2
noise = rng.randn(n_sample) * .3
y = x ** 3 - 0.5 * x ** 2 + noise

plt.scatter(x, y,  color='k', s=9)


# %%
def bootstrap_sample(x, y):
    bootstrap_indices = rng.choice(
        np.arange(y.shape[0]), size=y.shape[0], replace=True,
    )
    x_bootstrap_sample = x[bootstrap_indices]
    y_bootstrap_sample = y[bootstrap_indices]
    return x_bootstrap_sample, y_bootstrap_sample


# %%
n_bootstrap = 3
_, axs = plt.subplots(ncols=n_bootstrap, figsize=(16, 6))

for idx, (ax, _) in enumerate(zip(axs, range(n_bootstrap))):
    x_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(x, y)
    ax.scatter(
        x_bootstrap_sample, y_bootstrap_sample,
    )
    ax.set_title(f"Bootstrap sample #{idx}")

# %%
_, axs = plt.subplots(ncols=n_bootstrap, figsize=(16, 6))

forest = []
for idx, (ax, _) in enumerate(zip(axs, range(n_bootstrap))):
    x_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(x, y)
    ax.scatter(x_bootstrap_sample, y_bootstrap_sample)

    forest.append(
        DecisionTreeRegressor(max_depth=3, random_state=0).fit(
            x_bootstrap_sample.reshape(-1, 1), y_bootstrap_sample
        )
    )

    grid = np.linspace(np.min(x), np.max(x), num=300)
    y_pred = forest[-1].predict(grid.reshape(-1, 1))
    ax.plot(grid, y_pred, linewidth=3)

    ax.set_title(f"Fitted tree on boostrap sample #{idx}")

# %%

_, ax = plt.subplots()
ax.scatter(x, y)
y_pred_forest = []
for tree_idx, tree in enumerate(forest):
    y_pred = tree.predict(grid.reshape(-1, 1))
    ax.plot(
        grid, y_pred, "-.", label=f"Tree #{tree_idx} predictions", linewidth=3,
    )
    y_pred_forest.append(y_pred)

y_pred_forest = np.mean(y_pred_forest, axis=0)
ax.plot(
    grid, y_pred_forest, "--", label="Bagging predictions", linewidth=3,
)

plt.legend()

# %%
_, axs = plt.subplots(ncols=n_bootstrap, figsize=(16, 6))

random_forest = []
for idx, (ax, _) in enumerate(zip(axs, range(n_bootstrap))):
    x_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(x, y)
    ax.scatter(x_bootstrap_sample, y_bootstrap_sample)

    random_forest.append(
        DecisionTreeRegressor(
            max_depth=3, max_features="sqrt", random_state=0
            ).fit(
            x_bootstrap_sample.reshape(-1, 1), y_bootstrap_sample
        )
    )

    grid = np.linspace(np.min(x), np.max(x), num=300)
    y_pred = random_forest[-1].predict(grid.reshape(-1, 1))
    ax.plot(grid, y_pred, linewidth=3)

    ax.set_title(f"Fitted tree on boostrap sample #{idx}")

# %%

_, ax = plt.subplots()
ax.scatter(x, y)
y_pred_forest = []
for tree_idx, tree in enumerate(random_forest):
    y_pred = tree.predict(grid.reshape(-1, 1))
    ax.plot(
        grid, y_pred, "-.", label=f"Tree #{tree_idx} predictions", linewidth=3,
    )
    y_pred_forest.append(y_pred)

y_pred_forest = np.mean(y_pred_forest, axis=0)
ax.plot(
    grid, y_pred_forest, "--", label="Random forest predictions", linewidth=3,
)

plt.legend()


# %%
