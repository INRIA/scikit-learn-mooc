# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # K-means clustering with scikit-learn
#
# So far we have only addressed supervised learning models, namely regression
# and classification. In this module we introduce unsupervised learning for the
# first time.
#
# In this notebook we explore the k-means algorithm, which seeks to group data
# by a certain notion of similarity. To illustrate the different concepts, we
# use the Mall Customers dataset.
#
# Here we use clustering to group customers with similar profiles based on some
# characteristics, which then can be used for customer segmentation, and
# therefore, for designing better targeted campaigns.

# %%
import pandas as pd

data = pd.read_csv("../datasets/mall_customers.csv")
data

# %% [markdown]
# As we can see, this dataset includes the following information:
#
# - Gender: The gender of the customer.
# - Age: The age of the customer.
# - Annual Income (k$): The annual income of the customer (in thousands of
#   dollars).
# - Spending Score (1–100): The score ranges from 1 to 100, with a higher score
#   indicating a customer who spends more.
#
# In this case we cannot assign any of those columns to be the target. These are
# all features, each of them representing different aspects of a customer
# profile. We can verify that such features do not have a direct, predictable
# relationship with each other:

# %%
import seaborn as sns

_ = sns.pairplot(data, hue="Genre", height=4)

# %% [markdown]
# One could feel inclined to assigning labels to translate the task into a
# classification problem, instead of using clustering.
#
# One approach could be simple labeling: Low spenders, mid spenders, and high
# spenders (3 labels). But a priori nothing prevents us from defining multiple
# combinations, such as:
# - Young - Low spender - High income
# - Young adult - High spender - Mid income
# - Older adult - Mid spender - Low income
#
# We divided each numerical feature into 3 bins, leading to `3 ** n_features`
# possible combinations. But we could also have used different amounts of bins
# to define those labels and the problem rapidly becomes complex and subjective.
# The choice of how to bin or categorize customers features can introduce
# arbitrary boundaries, and the labels may not capture the nuances of customer
# behavior effectively or can lead to oversimplification. For some settings we
# rather let cluster labels emerge from the analysis, not from prior knowledge.
#
# Let's keep only the numerical values for the rest of this notebook. Having 3
# features is something we can still easily visualize.

# %%
data = data.drop(columns=["Genre"])

# %% [markdown]
# ## Training a k-means algorithm
#
# Intuitively, a good cluster should be compact (with points close to each
# other), dense (with a high concentration of data points), and well-separated
# from other clusters. In client segmentation, this means that different
# clusters should clearly represent well-defined differences in their profiles.
#
# First let's define a helper function to gain a visual intuition of the
# clusters as obtained provided a `model`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def plot_clusters(model, ax):

    cluster_labels = model.fit_predict(data)
    n_clusters = len(np.unique(cluster_labels))

    ax.scatter(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        data["Age"],
        c=cluster_labels,
        s=50,
        alpha=0.7,
    )
    ax.set_box_aspect(None, zoom=0.84)
    ax.set_xlabel("Annual Income (k$)", labelpad=15)
    ax.set_ylabel("Spending Score (1-100)", labelpad=15)
    ax.set_zlabel("Age", labelpad=15)
    ax.set_title(f"n_clusters={n_clusters}", y=0.99)
    _ = plt.tight_layout()


# %% [markdown]
# ```{tip}
# Here we used the `fit_predict` method, which does both steps at once: it
# learns from the data just as using `fit`, and immediately outputs labels (or
# cluster labels) to those same data points as would be the case using `predict`.
# ```
#
# In the plots below we use different numbers of clusters, by changing the
# hyperparameter `n_clusters`. Here the `random_state` controls the centroid
# initialization.

# %%
n_clusters_values = [2, 4, 6, 8]
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(17, 15), subplot_kw={"projection": "3d"}
)

for ax, n_clusters in zip(axes.flatten(), n_clusters_values):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    plot_clusters(model, ax)

# %% [markdown]
# In non-supervised learning, such as clustering, not having ground truth labels
# can make the model evaluation challenging. However, as we have discussed,
# it is still possible to define metrics that provide insight into the quality of
# the formed clusters.
#
# One common metric for evaluating clusters is Within-Cluster Sum of Squares
# (WCSS), also known as **inertia**, which measures how compact the clusters
# are. A lower WCSS indicates that the data points within each cluster are close
# to the cluster's centroid, suggesting that the cluster is well-formed.

# %%
wcss = []
n_clusters_values = range(2, 11)

for n_clusters in n_clusters_values:
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(data)
    wcss.append(model.inertia_)

plt.plot(n_clusters_values, wcss, marker="o")
plt.xlabel("Number of clusters (n_clusters)")
plt.ylabel("Inertia")
_ = plt.title("Elbow method using cluster inertia")

# %% [markdown]
# The so called elbow method can be subtile here, but it seems to match our
# visual intuition from the 3D plots: having 6 clusters seems to be the best
# choice for correctly identifying groups.
#
# Another useful metric is the Silhouette Score. A high silhouette score means
# that the data points are not only well-grouped within their own clusters but
# also well-separated from other clusters. A value of 0 indicates that the the
# decision boundary between two neighboring clusters may overlap, whereas
# negative values indicate that some samples might have been assigned to the
# wrong cluster.

# %%
from sklearn.metrics import silhouette_score

silhouette_scores = []
for n_clusters in n_clusters_values:
    model = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = model.fit_predict(data)
    score = silhouette_score(data, cluster_labels)
    silhouette_scores.append(score)

plt.plot(n_clusters_values, silhouette_scores, marker="o")
plt.xlabel("Number of clusters (n_clusters)")
plt.ylabel("Silhouette score")
_ = plt.title("Silhouette scores for different n_clusters")

# %% [markdown]
# The silhouette score reaches a maximum when `n_clusters=6`, which confirms
# both the visual intuition and the optimal number of clusters found using the
# elbow method.
