# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M4.01
#
# In this exercise we investigate the stability of the k-means algorithm. For
# such purpose, we use the RFM Dataset. RFM is a method used for analyzing
# customer value and the acronym RFM stands for the three dimensions:
#
# - Recency: How recently did the customer purchase;
# - Frequency: How often do they purchase;
# - Monetary Value: How much do they spend.
#
# It is commonly used in marketing and has received particular attention in
# retail and professional services industries as well.

# %%
import pandas as pd

data = pd.read_csv("../datasets/rfm_segmentation.csv")
data

# %% [markdown]
# As k-means clustering relies on computing distances between samples, in
# general we need to scale our data before training the clustering model. That
# was not the case in our previous notebook, as the features in the Mall
# customers dataset already have the same scale.
#
# Show that scaling is important or else "monetary" have a dominant impact when
# forming clusters. You can adapt the helper function `plot_clusters` from the
# previous notebook to make plots using `KMeans` for `n_clusters_values = [2, 4,
# 6, 8]` without scaling.

# %%
# solution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def plot_clusters(model, ax):

    cluster_labels = model.fit_predict(data)
    n_clusters = len(np.unique(cluster_labels))

    ax.scatter(
        data["monetary"],
        data["frequency"],
        data["recency"],
        c=cluster_labels,
        s=50,
        alpha=0.7,
    )
    ax.set_box_aspect(None, zoom=0.84)
    ax.set_xlabel("Monetary", labelpad=15)
    ax.set_ylabel("Frequency", labelpad=15)
    ax.set_zlabel("Recency", labelpad=15)
    ax.set_title(f"n_clusters={n_clusters}", y=0.99)
    _ = plt.tight_layout()


n_clusters_values = [2, 4, 6, 8]
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(17, 15), subplot_kw={"projection": "3d"}
)

for ax, n_clusters in zip(axes.flatten(), n_clusters_values):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    plot_clusters(model, ax)

# %%
# Create a pipeline composed by a `StandardScaler` followed by a `KMeans` step
# as the final predictor. Set the `random_state` for reproducibility. Then, make
# a plot of the WCSS or inertia for `n_clusters` varying from 2 to 10. You can
# use the following helper function for such purpose:

# %%
from sklearn.metrics import silhouette_score


def plot_n_clusters_scores(
    model,
    data,
    score_type="inertia",
    n_clusters_values=range(2, 11),
    alpha=1.0,
    title=None,
):
    """
    Plots clustering scores (inertia or silhouette) for a range of n_clusters.

    Parameters:
        model: A pipeline whose last step has a `n_clusters` hyperparameter.
        data: The input data to cluster.
        score_type: "inertia" or "silhouette" to decide which score to compute.
        n_clusters_values: Iterable of integers representing `n_clusters` to try.
        alpha: Transparency of the plot line, useful when several plots overlap.
        title: Optional title to set; default title used if None.
    """
    scores = []

    for n_clusters in n_clusters_values:
        model[-1].set_params(n_clusters=n_clusters)

        if score_type == "inertia":
            ylabel = "Inertia"
            model.fit(data)
            scores.append(model[-1].inertia_)
        elif score_type == "silhouette":
            ylabel = "Silhouette score"
            cluster_labels = model.fit_predict(data)
            data_transformed = model[:-1].transform(data)
            score = silhouette_score(data_transformed, cluster_labels)
            scores.append(score)
        else:
            raise ValueError(
                "score_type must be either 'inertia' or 'silhouette'"
            )

    plt.plot(n_clusters_values, scores, color="tab:blue", alpha=alpha)
    plt.xlabel("Number of clusters (n_clusters)")
    plt.ylabel(ylabel)
    _ = plt.title(title or f"{ylabel} for varying n_clusters", y=1.01)


# %%
# solution
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), KMeans(random_state=0))
model

# %%
# solution
plot_n_clusters_scores(model, data, score_type="inertia")

# %% [markdown]
# Let's check if the best choice of n_clusters remains stable when resampling
# the dataset. For such purpose:
# - Keep a fixed `random_state` for the `KMeans` step to isolate the
#   effect of data resampling.
# - Generate resamplings consisting of 90% of the data by using
#   `train_test_split` with `train_size=0.9`
# - Use the `plot_n_clusters_scores` function inside a `for` loop to make
#   multiple overlapping plots of the inertia, each time using a different
#   resamplings.
#
# Is the elbow (optimal number of clusters) stable across all different
# resamplings?

# %%
# solution
from sklearn.model_selection import train_test_split

for random_state in range(1, 11):
    data_subsample, _ = train_test_split(
        data, train_size=0.9, random_state=random_state
    )
    plot_n_clusters_scores(
        model,
        data_subsample,
        score_type="inertia",
        alpha=0.2,
        title="Stability of inertia across resamplings",
    )

# %% [markdown] tags=["solution"]
# The inertia changes drastically as a function of the subsamples, it is
# then not possible to systematically define an optimal number of clusters.

# %% [markdown]
# By default, `KMeans` uses a smart selection of the initial centroids called
# "k-means++". Instead of picking points completly at random, it tries several
# candidate centroids at each step and picks the best ones based on an
# estimation of how much they would help reduce the overall inertia. This method
# improves the chances of finding better cluster centroids and speeds up
# convergence compared to random initialization.
#
# Because "k-means++" already does a good job of finding suitable centroids, a
# single initialization is typically sufficient for most cases. That is why the
# parameter `n_init` in scikit-learn (which controls the number of times the
# algorithm is run with different centroid initializations) is set to 1 by
# default when `init="k-means++"`. Nevertheless, there may be cases (as when
# data is unevenly distributed) where increasing `n_init` may help ensuring a
# global minimal inertia.
#
# Repeat the previous example but setting `n_init=5`. Remeber to fix the
# `random_state` for the `KMeans` initialization to only estimate the
# variability related to resamplings of the data. Are the resulting inertia
# curves more stable?

# %%
# solution
model = make_pipeline(StandardScaler(), KMeans(n_init=5, random_state=0))
model

# %%
# solution
for random_state in range(1, 11):
    data_subsample, _ = train_test_split(
        data, train_size=0.9, random_state=random_state
    )
    plot_n_clusters_scores(
        model,
        data_subsample,
        score_type="inertia",
        alpha=0.2,
        title="Stability of inertia with n_init=5",
    )

# %% [markdown] tags=["solution"]
# The inertia is now stable, but an elbow is not clearly defined and then it is
# not possible to define an optimal number of clusters from this heuristics.

# %% [markdown]
# Repeat the experiment, but this time determine if the optimal number of
# clusters is stable across subsamplings when using the `silhouette_score`. Be
# aware that computing the silhouette score is more computationally costly than
# computing the inertia.

# %%
# solution
for random_state in range(1, 11):
    data_subsample, _ = train_test_split(
        data, train_size=0.9, random_state=random_state
    )
    plot_n_clusters_scores(
        model,
        data_subsample,
        score_type="silhouette",
        alpha=0.2,
        title="Stability of silhouette score with n_init=5",
    )

# %% [markdown] tags=["solution"]
# The silhouette score also varies as a function of the resampling, even after
# setting `n_init=5`. It may seem that the optimal number of clusters is
# sometimes 5, 6, or 9, depending on the sampling. We can conclude that in this
# case, what makes challenging to find the optimal number of clusters is not
# related to the metric, but possibly to the data itself, or to the modeling
# pipeline.

# %% [markdown]
# Once again repeat the experiment to determine the stability of the optimal
# number of clusters. This time, instead of using a `StandardScaler`, use a
# `QuantileTransformer` with default parameters as the preprocessing step in the
# pipeline. For the `KMeans` step, keep `n_init=5` and a fixed `random_state`.
# What happens in terms of silhouette score?

# %%
from sklearn.preprocessing import QuantileTransformer

model = make_pipeline(QuantileTransformer(), KMeans(n_init=5, random_state=0))
for random_state in range(1, 11):
    data_subsample, _ = train_test_split(
        data, train_size=0.9, random_state=random_state
    )
    plot_n_clusters_scores(
        model,
        data_subsample,
        score_type="silhouette",
        alpha=0.2,
        title="Stability of silhouette score\nwith n_init=5 and QuantileTransformer",
    )

# %% [markdown] tags=["solution"]
# The silhouette score is much more stable across resamplings. Moreover, the
# optimal number of clusters seems to be 2, as it provides the highest score,
# indicating that the data points are well-separated and correctly grouped with
# good cohesion. However, 4 or 6 clusters may still make sense if the clustering
# has specific use cases or domain relevance, but you should be cautious as the
# relatively low silhouette score suggests that some points may be misassigned.
#
# Indeed, we can plot the transformed space and observe some samples seem to be
# misassigned.

# %% tags=["solution"]
model = make_pipeline(
    QuantileTransformer(), KMeans(n_init=5, n_clusters=6, random_state=0)
)
model

# %% tags=["solution"]
cluster_labels = model.fit_predict(data)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
scatter = ax.scatter(
    *model[:-1].transform(data).T,
    c=cluster_labels,
    cmap="viridis",
    s=50,
    alpha=0.7,
)
ax.set_box_aspect(None, zoom=0.84)
ax.set_xlabel("Transformed Monetary", labelpad=15)
ax.set_ylabel("Transformed Frequency", labelpad=15)
ax.set_zlabel("Transformed Recency", labelpad=15)
ax.set_title("Clusters in quantile-transformed space", y=0.99)
_ = plt.tight_layout()

# %% [markdown] tags=["solution"]
# From the plot above, it would feel more natural to cluster together all
# samples with transformed monetary equal to 0 instead of dividing it into 2
# different clusters. Moreover, those clusters seem to take samples with low but
# non-zero values of monetary, that would belong more naturally to other
# clusters.
#
# What happens here is that data points at transformed monetary equal to 0 are
# not isotropic in the 3 dimensions, i.e. they are spread on a plane of
# transformed recency and frequency ranging from 0 to 1. Remember that k-means
# consists of minimizing samples' euclidean distances to their assigned
# centroid. As a consequence, k-means is more appropriate for clusters that are
# isotropic and normally distributed.
#
# We will learn more about how to deal with anisotropic clusters in a future
# notebook.
