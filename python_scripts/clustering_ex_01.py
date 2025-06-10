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
# # 📝 Exercise M4.01
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
# retail and professional services industries as well. Here we subsample the
# dataset to ease the calculations.

# %%
import pandas as pd

data = pd.read_csv("../datasets/rfm_segmentation.csv")
data = data.sample(n=2000, random_state=0).reset_index(drop=True)
data

# %% [markdown]
# We can explore the data using a seaborn `pairplot`.

# %%
import seaborn as sns

_ = sns.pairplot(data)

# %% [markdown]
# As k-means clustering relies on computing distances between samples, in
# general we need to scale our data before training the clustering model.
#
# Modify the color of the `pairplot` to represent the cluster labels as
# predicted by `KMeans` without any scaling. Try different values for
# `n_clusters`, for instance, `n_clusters_values=[2, 3, 4]`. Do all features
# contribute equally to forming the clusters in their original scale?

# %%
# Write your code here.

# %% [markdown]
# Create a pipeline composed by a `StandardScaler` followed by a `KMeans` step
# as the final predictor. Set the `random_state` of `KMeans` for
# reproducibility. Then, make a plot of the WCSS or inertia for `n_clusters`
# varying from 1 to 10. You can use the following helper function for such
# purpose:

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def plot_n_clusters_scores(
    model,
    data,
    score_type="inertia",
    n_clusters_values=None,
    alpha=1.0,
    title=None,
):
    """
    Plots clustering scores (inertia or silhouette) for a range of n_clusters.

    Parameters:
        model: A pipeline whose last step has a `n_clusters` hyperparameter.
        data: The input data to cluster.
        score_type: "inertia" or "silhouette" to decide which score to compute.
        alpha: Transparency of the plot line, useful when several plots overlap.
        title: Optional title to set; default title used if None.
    """
    scores = []

    if n_clusters_values is None:
        if score_type == "inertia":
            n_clusters_values = range(1, 11)
        else:
            n_clusters_values = range(2, 11)

    for n_clusters in n_clusters_values:
        model[-1].set_params(n_clusters=n_clusters)
        if score_type == "inertia":
            ylabel = "WCSS (inertia)"
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
# Write your code here.

# %% [markdown]
# Let's see if we can find one or more stable candidates for `n_clusters` using
# the elbow method when resampling the dataset. For such purpose:
# - Generate randomly resampled data consisting of 50% of the data by using
#   `train_test_split` with `train_size=0.5`. Changing the `random_state` to do
#   the split leads to different samples.
# - Use the `plot_n_clusters_scores` function inside a `for` loop to make
#   multiple overlapping plots of the inertia, each time using a different
#   resampling. 10 resampling iterations should be enough to draw conclusions.
# - You can choose to set the `random_state` value of the `KMeans` step, but be
#   aware that even if we fix `random_state=0` in all resampling iterations,
#   k-means will still choose different initial centroids for different data
#   samples, so fixing it or not should not change the conclusions w.r.t. to
#   stability to resampling.
#
# Is the elbow (optimal number of clusters) stable when resampling?

# %%
# Write your code here.

# %% [markdown]
# By default, `KMeans` uses a smart selection of the initial centroids called
# "k-means++". Instead of picking points completely at random, it tries several
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
# Repeat the previous example but setting `n_init=5`. Remember to fix the
# `random_state` for the `KMeans` initialization to only estimate the
# variability related to the resampling of the data. Are the resulting inertia
# curves more stable?

# %%
# Write your code here.

# %% [markdown]
# Repeat the experiment, but this time determine if the optimal number of
# clusters (with `StandarScaler` and `n_init=5`) is stable across subsamplings
# in terms of the `silhouette_score`. Be aware that computing the silhouette
# score is more computationally costly than computing the inertia.

# %%
# Write your code here.

# %% [markdown]
#
# Once again repeat the experiment to determine the stability of the optimal
# number of clusters. This time, instead of using a `StandardScaler`, use a
# `QuantileTransformer` with default parameters as the preprocessing step in
# the pipeline. Contrary to `StandardScaler`, `QuantileTransformer` is a
# nonlinear transformation that maps the features with a long tail
# distributions to a uniform distribution, which is the case for the
# "frequency" and "monetary" features in the RFM dataset.
#
# For the `KMeans` step, keep `n_init=5`.
#
# What happens in terms of silhouette score? Does this make it possible to
# identify stable and qualitatively interesting clusters in this data?

# %%
# Write your code here.
