# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Non-convex clustering using HDBSCAN
#
# We have previously mentioned that k-means consists of minimizing the samples
# euclidean distances to their assigned centroid. As a consequence, k-means is
# more appropriate for clusters that are isotropic and normally distributed
# (look like blobs). In this notebook we introduce another clustering technique
# named HDBSCAN, an acronym which stands for "Hierarchical Density-Based Spatial
# Clustering of Applications with Noise".
#
# Let's explain each of those tearms. HDBSCAN is hierarchical, which means it
# handles data with clusters nested within each other. The user controls the
# level in the hierarchy at which clusters are formed.
#
# It is density-based (and therefore non-parametric, contrary to K-means)
# because it does not assume a specific shape or number of clusters. Instead, it
# automatically finds the clusters based on areas where data points are densely
# packed together. In other words, it looks for regions of high density (many
# data points close to each other) and forms clusters around them. This allows
# it to find clusters of varying shapes and sizes.
#
# HDBSCAN assigns a label of -1 to points that do not have enough neighbors (low
# density) to be considered part of a cluster or are too far from any dense
# region (too isolated from core points). They are usually considered to be
# noise.
#
# Let's first illustrate those concepts with a toy dataset.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

rng = np.random.default_rng(1)

centers = np.array([[-4.8, 2.0], [-3.5, -4.5]])
X_gaussian, _ = make_blobs(
    n_samples=[200, 60],
    centers=centers,
    cluster_std=[1.0, 0.5],
    random_state=42,
)

# Two anisotropic blobs
centers = np.array([[1.0, 5.1], [3.0, 0.9]])
X_aniso_base, y_aniso_base = make_blobs(
    n_samples=200, centers=centers, random_state=0
)

# Define two different transformations
transformation_0 = np.array([[0.6, -0.6], [-0.4, 0.8]])
transformation_1 = np.array([[1.5, 0], [0, 0.3]])

# Apply different transformations to each blob
X_aniso = np.copy(X_aniso_base)
X_aniso[y_aniso_base == 0] = np.dot(
    X_aniso_base[y_aniso_base == 0], transformation_0
)
X_aniso[y_aniso_base == 1] = np.dot(
    X_aniso_base[y_aniso_base == 1], transformation_1
)


def make_wavy_blob(n_samples, shift=0.0, noise=0.2, freq=3):
    "Make wavy blobs in feature space"
    x = np.linspace(-3, 3, n_samples)
    y = np.sin(freq * x) + shift
    x += rng.normal(scale=noise, size=n_samples)
    y += rng.normal(scale=noise, size=n_samples)
    return np.vstack((x, y)).T


X_wave1 = make_wavy_blob(100, shift=4.7, freq=1)
transformation = np.array([[0.6, -0.6], [0.4, 0.8]])
X_wave1 = np.dot(X_wave1, transformation)
X_wave2 = make_wavy_blob(200, shift=-2.0, freq=2)


X_noise = rng.uniform(low=-8, high=8, size=(100, 2))  # background noise

X_all = np.vstack((X_gaussian, X_aniso, X_wave1, X_wave2, X_noise))

plt.scatter(X_all[:, 0], X_all[:, 1], alpha=0.6)
_ = plt.title("Synthetic dataset")

# %% [markdown]
# Let's first try to find a cluster structure using K-means with 6 clusters to
# match our data generating process.

# %%
from sklearn.cluster import KMeans

cluster_labels = KMeans(n_clusters=6, random_state=0).fit_predict(X_all)
_ = plt.scatter(X_all[:, 0], X_all[:, 1], c=cluster_labels, alpha=0.6)

# %% [markdown]
# We could try to increase the number of clusters to avoid grouping
# unrelated points in the same cluster:

# %%
cluster_labels = KMeans(n_clusters=10, random_state=0).fit_predict(X_all)
_ = plt.scatter(X_all[:, 0], X_all[:, 1], c=cluster_labels, alpha=0.6)

# %% [markdown]
# However, we can observe that too many cluster divides the high density region
# too much, while it continues grouping unrelated points together. Therefore,
# adjusting the number of clusters is not enough to get good results in this
# kind of data.
#
# Let's now repeat the experiment using HDBSCAN instead. For this clustering
# technique, the most important hyperparameter is `min_cluster_size`, which
# controls the minimum number of samples for a group to be considered a cluster;
# groupings smaller than this size will be left as noise.

# %%
from sklearn.cluster import HDBSCAN

cluster_labels = HDBSCAN(min_cluster_size=10).fit_predict(X_all)
_ = plt.scatter(X_all[:, 0], X_all[:, 1], c=cluster_labels, alpha=0.6)

# %% [markdown]
# Let's now apply HDBSCAN to a more realistic use-case: the geospatial columns
# of the California Housing Dataset.

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)

# %% [markdown]
# We can use plotly to first visualize the housing prices across the state of
# California.

# %%
import plotly.express as px


def plot_map(df, color_feature):
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color=color_feature,
        zoom=5,
        height=600,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={
            "lat": df["Latitude"].mean(),
            "lon": df["Longitude"].mean(),
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig


fig = plot_map(data, target)
fig

# %% [markdown]
# We can first use K-means to group data points into different spatial regions.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

geo_columns = ["Latitude", "Longitude"]
geo_data = data[geo_columns]

kmeans_pipeline = make_pipeline(
    StandardScaler(), KMeans(n_clusters=20, random_state=0)
)

cluster_labels = kmeans_pipeline.fit_predict(geo_data)
cluster_labels

# %%
fig = plot_map(data, cluster_labels.astype("str"))
fig

# %% [markdown]
# We can observe that results are really influenced by the K-means that favors
# "blobby"-shaped clusters. Let's try again with HDBSCAN which should not suffer
# from the same bias.

# %%
from sklearn.cluster import HDBSCAN

hdbscan_pipeline = make_pipeline(HDBSCAN(min_cluster_size=100))

cluster_labels = hdbscan_pipeline.fit_predict(geo_data)
cluster_labels

# %%
fig = plot_map(data, cluster_labels.astype("str"))
fig

# %% [markdown]
# HDBSCAN automatically detect highly populated areas that match urban centers,
# potentially increasing the housing prices. In addition we observe that points
# lying in low density regions are labeled `-1` instead of being forced into a
# cluster.
#
# The number of resulting clusters is a consequence of the choice of
# `min_cluster_size`:

# %%
len(np.unique(cluster_labels))

# %% [markdown]
# Decreasing `min_cluster_size` will increase the number of clusters:

# %%
hdbscan_pipeline = make_pipeline(HDBSCAN(min_cluster_size=30))
cluster_labels = hdbscan_pipeline.fit_predict(geo_data)
fig = plot_map(data, cluster_labels.astype("str"))
fig

# %%
len(np.unique(cluster_labels))
