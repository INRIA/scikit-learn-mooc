# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # K-means clustering
#
# So far we have only addressed supervised learning models, namely regression
# and classification. In this module we introduce unsupervised learning for the
# first time.
#
# In this notebook we explore the k-means algorithm, which seeks to group data
# based on the pairwise distances between data points. To illustrate the
# different concepts, we will extract some numerical features from the penguins
# dataset.

# %%
import pandas as pd

columns_to_keep = [
    "Species",
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
]
penguins = pd.read_csv("../datasets/penguins.csv")[columns_to_keep].dropna()
penguins

# %% [markdown]
# We know that this datasets contains data about 3 different species of
# penguins, but we will not explicitly rely on this information and instead
# treat the problem as an unsupervised data analysis task. The goal is to
# assess whether K-means can help us discover meaningful clusters in the data.
#
# Let's hide this column for now. We will only use it at the end of the notebook:
species = penguins["Species"]
penguins = penguins.drop(columns=["Species"])

# %% [markdown]
#
# Let's take a first look at the structure of the numerical features using a
# pairplot:

# %%
import seaborn as sns

_ = sns.pairplot(penguins, height=4)

# %% [markdown]
#
# On these plots, we more or less easily visually recognize 2 to 3 clusters
# depending on the feature pairs.
#
# We suspect that the clusters overlap because female penguins are generally
# smaller than male penguins:

# %%
_ = sns.pairplot(penguins, hue="Sex", height=4)

# %% [markdown]
#
# Let us focus on female individuals to visually assess if the clusters are
# better separated:

# %%
female_penguins = penguins.query("Sex == 'FEMALE'")
_ = sns.pairplot(female_penguins, height=4)

# %% [markdown]
#
# As we can see, the clusters look better separated on this subset of the
# dataset.
#
# In particular we can see that if we only consider:
# - **Culmen Length** and **Body Mass**, we can distinguish 3 clusters;
# - **Culmen Depth** and **Body Mass**, we can distinguish 2 clusters.
#
# Let's try to apply the k-means algorithm on the first pairs of columns to see
# whether we can find the clusters that we visually identified.

# %%
from sklearn.cluster import KMeans

kmeans_cl_vs_bm = KMeans(n_clusters=3, random_state=0)
kmeans_labels_cd_vs_bm = kmeans_cl_vs_bm.fit_predict(
    female_penguins[["Culmen Length (mm)", "Body Mass (g)"]]
)
kmeans_labels_cd_vs_bm

# %% [markdown]
#
# The `fit_predict` method returns the cluster labels for each data point coded
# with an arbitrary integer between 0 and `n_clusters - 1`.
#
# Let's consolidate these labels in the original dataframe and visualize the
# clusters:

# %%
clustered_female_peng = female_penguins.copy()
ax = sns.scatterplot(
    data=female_penguins.assign(kmeans_labels=kmeans_labels_cd_vs_bm),
    x="Culmen Length (mm)",
    y="Body Mass (g)",
    hue="kmeans_labels",
    palette="deep",
    alpha=0.7,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %% [markdown]
#
# The result is disappointing: the 3 clusters found by k-means do not match
# what we would have naively expected from the scatter plot.
#
# What could explain this?
#
# Clusters are defined by the distance between data points, and the `KMeans`
# algorithm tries to minimize the distance between data points and their
# cluster centroid. But as we can see on the axis of the scatter plot, the
# values of "Culmen Length (mm)" and "Body Mass (g)" are not on the same scale.
#
# If we use the original units, the distances between data points are almost
# entirely dominated by the "Body Mass (g)" feature,  which has numerical
# values expressed on a scale that is much larger than the "Culmen Length (mm)"
# feature.
#
# We can visualize this by plotting the data by disabling the automated visual
# scaling of the axes by manually setting the same numerical limits for both
# axes:

# %%
min_numerical_value = 0
max_numerical_value = clustered_female_peng["Body Mass (g)"].max() * 1.1
ax = sns.scatterplot(
    data=clustered_female_peng,
    x="Culmen Length (mm)",
    y="Body Mass (g)",
    hue="K-means label",
    palette="deep",
    alpha=0.7,
)
ax.set(
    xlim=(min_numerical_value, max_numerical_value),
    ylim=(min_numerical_value, max_numerical_value),
    aspect="equal",
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %% [markdown]
#
# Under this new perspective, the k-means clustering results make more sense:
# the "Culmen Length" is not taken into account because the numerical values
# expressed in mm are much smaller than the "Body Mass" values expressed in
# grams.
#
# To mitigate this problem, we can instead define a pipeline to use always
# standardize the values of the numerical features before applying the
# clustering algorithm. This way, all features will have the same scale and
# contribute more or less equally to the distance calculations.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scaled_kmeans_cl_vs_bm = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, random_state=0),
)
scaled_kmeans_labels_cd_vs_bm = scaled_kmeans_cl_vs_bm.fit_predict(
    female_penguins[["Culmen Length (mm)", "Body Mass (g)"]]
)
clustered_female_peng = female_penguins.copy()
clustered_female_peng["K-means label"] = scaled_kmeans_labels_cd_vs_bm
ax = sns.scatterplot(
    data=clustered_female_peng,
    x="Culmen Length (mm)",
    y="Body Mass (g)",
    hue="K-means label",
    palette="deep",
    alpha=0.7,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %% [markdown]
#
# Now the results of the k-means cluster better match our visual intuition on
# this pair of features.
#
# Let's do a similar analysis on the second pair of features, namely "Culmen
# Depth (mm)" and "Body Mass (g)". To do so, let's refactor the code above as a
# utility function:


# %%
def plot_kmeans_clusters_on_2d_data(
    clustering_model,
    data,
    first_feature_name,
    second_feature_name,
):
    labels = clustering_model.fit_predict(
        data[[first_feature_name, second_feature_name]]
    )
    ax = sns.scatterplot(
        data=data.assign(kmeans_labels=labels),
        x=first_feature_name,
        y=second_feature_name,
        hue="kmeans_labels",
        palette="deep",
        alpha=0.7,
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


plot_kmeans_clusters_on_2d_data(
    make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=2, random_state=0),
    ),
    female_penguins,
    "Culmen Depth (mm)",
    "Body Mass (g)",
)

# %% [markdown]
#
# Here again the clusters are well separated and the k-means algorithm
# identified clusters that match our visual intuition.
#
# We can also try to apply the k-means algorithm with a larger value for
# `n_clusters`:

# %%
plot_kmeans_clusters_on_2d_data(
    make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=6, random_state=0),
    ),
    female_penguins,
    "Culmen Length (mm)",
    "Body Mass (g)",
)

# %% [markdown]
#
# When we select a large value of `n_clusters`, we observe that k-means will
# build as many groups as requested even if the resulting groups are not well
# separated.
#
# Let's now see if can use this intuition on cluster separation to identify
# suitable values for the number of clusters based on heuristic methods
# introduced earlier in the course.
#
# Let's start by plotting the evolution of the WCSS (Within-Cluster Sum of
# Squares) metric as a function of the number of clusters.

# %%
import matplotlib.pyplot as plt

wcss = []
n_clusters_values = range(1, 11)

for n_clusters in n_clusters_values:
    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=n_clusters, random_state=0),
    )
    cluster_labels = model.fit_predict(
        female_penguins[["Culmen Length (mm)", "Body Mass (g)"]]
    )
    wcss.append(model.named_steps["kmeans"].inertia_)

plt.plot(n_clusters_values, wcss, marker="o")
plt.xlabel("Number of clusters (n_clusters)")
plt.ylabel("WCSS (or inertia)")
_ = plt.title("Elbow method using WCSS")

# %% [markdown]
#
# As expected the WCSS value decreases as the number of clusters increases and
# we can observe a so-called "elbow" in the curve (the point with maximum
# curvature) around `n_clusters=3`. This matches the number of cluster found by
# our visual intuition when looking at this 2D scatter plots.
#
# However, the elbow method is not always easy to read.
#
# Let's try to use the silhouette score instead. Note that this method requires
# access to the preprocessed features:

# %%
from sklearn.metrics import silhouette_score


def plot_silhouette_scores(
    data,
    clustering_model=None,
    preprocessor=None,
    n_clusters_values=range(2, 11),
):
    if clustering_model is None:
        clustering_model = KMeans(random_state=0)

    if preprocessor is None:
        preprocessor = StandardScaler()

    preprocessed_data = preprocessor.fit_transform(data)

    silhouette_scores = []
    for n_clusters in n_clusters_values:
        clustering_model.set_params(n_clusters=n_clusters)
        cluster_labels = clustering_model.fit_predict(preprocessed_data)
        score = silhouette_score(preprocessed_data, cluster_labels)
        silhouette_scores.append(score)

    plt.plot(n_clusters_values, silhouette_scores, marker="o")
    plt.xlabel("Number of clusters (n_clusters)")
    plt.ylabel("Silhouette score")
    _ = plt.title("Silhouette scores for different n_clusters")


plot_silhouette_scores(
    female_penguins[["Culmen Length (mm)", "Body Mass (g)"]],
)

# %% [markdown]
#
# The silhouette score reaches a maximum when `n_clusters=3`, which confirms
# our visual intuition on this 2D dataset.
#
# We can also notice that the silhouette score is also very high for
# `n_clusters=2` and has an intermediate value for `n_clusters=4`. It's
# possible that those two values would also yield qualitatively meaningful
# clusters, but this probably not the case for `n_clusters=5` or larger.
#
# Let's compare this to the results obtained on the second pair of features:

# %%
plot_silhouette_scores(
    female_penguins[["Culmen Depth (mm)", "Body Mass (g)"]],
)

# %% [markdown]
#
# For this feature set, the plot clearly shows that the silhouette score
# reaches a maximum when `n_clusters=2`, which matches our visual intuition
# from the scatter plot of this 2D feature set.

# %% [markdown]
#
# We can now try to apply the k-means algorithm on the full dataset, i.e. on
# all numerical features and all the rows, to see whether k-means can discover
# meaningful clusters in the data automatically.
#
# We also include the `Sex` feature in the clustering model to see whether
# it can help the algorithm to find better clusters.

# %%
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder

preprocessor = make_column_transformer(
    (
        OneHotEncoder(drop="if_binary"),
        make_column_selector(dtype_exclude="number"),
    ),
    (
        StandardScaler(),
        make_column_selector(dtype_include="number"),
    ),
)
plot_silhouette_scores(penguins, preprocessor=preprocessor)

# %% [markdown]
#
# Based on the silhouette scores, it seems that k-means would prefer to cluster
# those features into either 2 or 6 clusters.
#
# Let's try to visualize the clusters obtained with `n_clusters=6`:

# %%

model = make_pipeline(
    preprocessor,
    KMeans(n_clusters=6, random_state=0),
)
cluster_labels = model.fit_predict(penguins)
_ = sns.pairplot(
    penguins.assign(cluster_label=cluster_labels),
    hue="cluster_label",
    palette="deep",
    height=4,
)

# %% [markdown]
#
# Since this is high-dimensional data (5D), the pairplot (computed only for the
# 4 numerical features) only offers a limited perspective on the clusters.
# Despite this limitation, the clusters do appear meaningful, and in particular
# we can notice that they could potentially correspond to the 3 species of
# penguins present in the dataset (Adelie, Chinstrap, and Gentoo) further
# splitted by Sex (2 clusters for each species, one for males and one for
# females).
#
# Let's try to confirm this hypothesis by looking at the original "Species"
# labels combined with the "Sex":

# %%
species_and_sex_labels = species + " " + penguins["Sex"]
species_and_sex_labels.value_counts()

# %%
_ = sns.pairplot(
    penguins.assign(species_and_sex=species_and_sex_labels),
    hue="species_and_sex",
    palette="deep",
    height=4,
)

# %% [markdown]
#
# This plot seems to be very similar to the pairplot we obtained with the 6
# clusters found by k-means on our preprocessed data. Note that the colors are
# different, because the ordering of the labels is arbitrary (both for the
# k-means cluster and the manually assigned labels). But the way of grouping
# the data points look similar.
#
# Let's quantify the agreement between the clusters found by k-means and the
# combination of the "Species" and "Sex" labels using the [Normalized Mutual
# Information](https://scikit-learn.org/stable/modules/clustering.html#mutual-info-score)
# (NMI) score.

# %%
from sklearn.metrics.cluster import normalized_mutual_info_score

nmi = normalized_mutual_info_score(
    species_and_sex_labels,
    cluster_labels,
)
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

# %% [markdown]
#
# This value is very close to 1.0, which indicates a very strong agreement.
#
# The conclusion is that we relate the clusters found by running k-means on
# those preprocessed features to a meaningful (human) way to partition the
# penguins records.
#
# Note however that this is not always the case. For **k-means to yield
# meaningful results, the data must be have an approximately balanced, convex
# and isotropic cluster structure** after preprocessing. That is, the clusters
# must have a spherical shape in the feature space and approximately the same
# size.
#
# We cannot stress enough that the choice of the features and preprocessing
# steps are crucial: if we had not standardized the numerical data, or we had
# not included the "Sex" feature or if we had scaled its one-hot encoding by a
# factor of 10, we would probably not have been able to discover interpretable
# clusters.
#
# Furthermore, **many natural datasets would not satisfy the k-means
# assumptions** even after non-trivial preprocessing. In those cases, we can
# either try alternatives to k-means that favor different cluster shapes (for
# instance HDBSCAN or Gaussian Mixture Models) or we can try to isolate
# row-wise or column-wise subsets of the data that are more likely to exhibit a
# cluster structure. Or sometimes, we can decide to partition the data with
# k-means with a large number of clusters, even if they are not interpretable
# and use the distance to centroids as preprocessing for another task. It all
# depends on the specific application domain and the downstream use of the
# resulting clusters.
#
# Finally, notice that we used extra supervised information to quantitatively
# assess the quality of the match between the clusters found by k-means and our
# interpretation. In practice, this is often impossible, as we do not have
# access to human assigned labels for each row in the data. Or, if we have, we
# might want to use them to train the clustering model, but instead we would
# rather use them as the target variable to train a supervised classifier.
