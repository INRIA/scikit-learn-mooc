# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Making clusters part of a supervised pipeline
#
# This notebook explores how K-means clustering can be used as a feature
# engineering step to improve the performance of a regression model.
#
# Here we use the California Housing dataset, which includes information about
# the geographic location (latitude and longitude).
#
# Our goal is to predict the median house value (MedHouseVal) using a ridge
# regression model, to investigate whether adding features derived from applying
# K-means to geographic coordinates can improve the pipeline's predictive
# performance.

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

# %% [markdown]
# We can first design a predictive pipeline that completly ignores the
# coordinates:

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import make_column_transformer

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=0
)
geo_columns = ["Latitude", "Longitude"]
model_drop_geo = make_pipeline(
    make_column_transformer(("drop", geo_columns), remainder="passthrough"),
    StandardScaler(),
    Ridge(alpha=1e-12),
)
test_error_drop_geo = -cross_val_score(
    model_drop_geo, data_train, target_train, scoring="neg_mean_absolute_error"
)
print(
    "The test MAE without geographical features is: "
    f"{test_error_drop_geo.mean():.2f} ± {test_error_drop_geo.std():.2f} k$"
)

# %% [markdown]
# We observe a Mean Absolute Error of approximately 57k$ when dropping the
# geographical features.
#
# As seen in the previous notebook, we suspect that the price information may be
# linked to the distance to the nearest urban center, and proximity to the
# coast:

# %%
import plotly.express as px

fig = px.scatter_map(
    data,
    lat="Latitude",
    lon="Longitude",
    color=target,
    zoom=5,
    height=600,
    labels={"color": "price (k$)"},
)
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={
        "lat": data["Latitude"].mean(),
        "lon": data["Longitude"].mean(),
    },
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
)
fig.show(renderer="notebook")

# %% [markdown]
# We first feed the coordinates directly to the linear model without
# transformation.

# %%
model_naive_geo = make_pipeline(StandardScaler(), Ridge(alpha=1e-12))
test_error_naive_geo = -cross_val_score(
    model_naive_geo,
    data_train,
    target_train,
    scoring="neg_mean_absolute_error",
)
print(
    "The test MAE with raw geographical features is: "
    f"{test_error_naive_geo.mean():.2f} ± {test_error_naive_geo.std():.2f} k$"
)

# %% [markdown]
# Including the geospatial data naively improves the performance a bit, however,
# we suspect that we can do better by introducing features that represent
# proximity to points of interests (urban centers, the coast, parks, etc.).
#
# We could look for a dataset containing all the coordinates of the city
# centers, the coast line and other points of interest in California, then
# manually engineer such features. However this would require a non-trivial
# amount of code. Instead we can rely on K-means to achieve something similar
# implicitly: we use it to find a large number of centroids from the housing
# data directly and consider each centroid a potential point of interest.
#
# The `KMeans` class implements a `transform` method that, given a set of data
# points as an argument, computes the distance to the nearest centroid for each
# of them. As a result, `KMeans` can be used as a preprocessing step in a
# feature engineering pipeline as follows:

# %%
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

model_cluster_geo = make_pipeline(
    make_column_transformer(
        (KMeans(n_clusters=100), geo_columns),
        remainder="passthrough",
        force_int_remainder_cols=False,  # Silence a warning in scikit-learn v1.6
    ),
    StandardScaler(),
    Ridge(alpha=1e-12),
)
test_error_cluster_geo = -cross_val_score(
    model_cluster_geo,
    data_train,
    target_train,
    scoring="neg_mean_absolute_error",
)
print(
    "The test MAE with clustered geographical features is:"
    f" {test_error_cluster_geo.mean():.2f} ±"
    f" {test_error_cluster_geo.std():.2f} k$"
)

# %% [markdown]
# The resulting mean test error is much lower. Furthermore, `KMeans` is now part
# of a supervised pipeline, which means we can use a grid-search to tune
# `n_clusters` as we have learned in previous modules of this course.

# %%
from sklearn.model_selection import GridSearchCV

param_name = "columntransformer__kmeans__n_clusters"
param_grid = {param_name: [10, 30, 100, 300, 1_000, 3_000]}
grid_search = GridSearchCV(
    model_cluster_geo, param_grid=param_grid, scoring="neg_mean_absolute_error"
)
grid_search.fit(data_train, target_train)

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
grid_search_results = pd.DataFrame(grid_search.cv_results_)[results_columns]
grid_search_results["mean_test_error"] = -grid_search_results[
    "mean_test_score"
]
grid_search_results = (
    grid_search_results.drop(columns=["mean_test_score"])
    .rename(columns={"param_" + param_name: "n_clusters"})
    .round(3)
)
grid_search_results.sort_values("mean_test_error", ascending=False)

# %% [markdown]
# Larger number of clusters increases the predictive performance at the cost
# of larger fitting and prediction times.

# %%
labels = {
    "mean_fit_time": "CV fit time (s)",
    "mean_test_error": "CV score (MAE)",
}
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
# We can finally evaluate the best model found by our analysis and see how well
# it can generalize.

# %%
best_n_clusters = grid_search.best_params_[param_name]
print(
    f"The test MAE with {best_n_clusters} clusters is: "
    f"{-grid_search.score(data_test, target_test):.3f} k$"
)

# %% [markdown]
# This notebook demonstrates one way to leverage clustering for non-linear
# feature engineering, but there are many ways to compose unsupervised models in
# a supervised-learning pipeline.
#
# We can finally observe that even if K-means was not the best clustering
# algorithm from a qualitatively point of view (as presented in the previous
# notebook), it is still helpful at crafting predictive features.
