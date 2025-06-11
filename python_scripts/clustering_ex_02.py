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
# # 📝 Exercise M4.02
#
# In a previous notebook we introduced the use of performance metrics to
# evaluate a clustering model when we have access to labeled data, namely the
# V-measure and Adjusted Rand Index (ARI). In this exercise you will get
# familiar with another supervised metric for clustering, known as Adjusted
# Mutual Information (AMI).
#
# To illustrate the different concepts, we retain some of the features from the
# penguins dataset.

# %%
import pandas as pd

columns_to_keep = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
    "Species",
]
penguins = pd.read_csv("../datasets/penguins.csv")[columns_to_keep].dropna()
species = penguins["Species"].str.split(" ").str[0]
penguins = penguins.drop(columns=["Species"])
penguins

# %% [markdown]
# We recall that the silhouette score presented a maximum when `n_clusters=6`
# when using all of the features above (not the species). Our hypothesis was
# that those clusters correspond to the 3 species of penguins present in the
# dataset (Adelie, Chinstrap, and Gentoo) further splitted by Sex (2 clusters
# for each species).
#
# Repeat the same pipeline consisting of a `OneHotEncoder` with
# `drop="if_binary"` for the "Sex" column, a `StandardScaler` for the other
# columns. The final estimator should be `KMeans` with `n_clusters=6`. You can
# set the `random_state` for reproducibility, but that should not change the
# interpretation of the results.

# %%
# Write your code here.

# %% [markdown]
# Make two `sns.scatterplot` of "Culmen Length (mm)" versus "Flipper Length
# (mm)", side-by-side. On one of them, the `hue` should be the "species and sex"
# coming from the known information in the dataset, and the `hue` in the other
# should be the cluster labels.
#
# Only the colors may differ, as the ordering of the labels is arbitrary (both
# for the k-means cluster and the "true" labels).

# %%
# Write your code here.

# %% [markdown]
# We now have a visual intuition of the agreement between the clusters found by
# k-means and the combination of the "Species" and "Sex" labels. We can further
# quantify it using the Adjusted Mutual Information (AMI) score.
#
# Use
# [`sklearn.metrics.adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)
# to compare both sets of labels. The AMI returns a value of 1 when the two
# partitions are identical (ie perfectly matched)

# %%
# Write your code here.

# %% [markdown]
# Now use a
# [`sklearn.preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
# to `fit_transform` the "true" labels (coming from combinations of species and
# sex). What would be the accuracy if we tried to use it to measure the
# agreement between both sets of labels?

# %%
# Write your code here.

# %% [markdown]
# Permute the cluster labels using `np.random.permutation`, then compute both
# the AMI and the accuracy when comparing the true and permuted labels. Are they
# sensitive to relabeling?

# %%
# Write your code here.

# %% [markdown]
# AMI is designed to return a value near zero (it can be negative) when the
# clustering is no better than random.
#
# To understand how AMI corrects for chance, compare the true labels with a
# completely random labeling using `np.random.randint` to generate as many
# labels as rows in the dataset, each containing a value between 0 and 5 (to
# match the number of clusters).

# %%
# Write your code here.

# %% [markdown]
#
# We can conclude by comparing AMI to other metrics:
#
# - Adjusted Rand Index (ARI): Also corrects for chance, but it counts pairs of
#   points, in other words, how many pairs that are together in the true labels
#   are also together in the clusters. It is combinatorial, not based on
#   information-theory as AMI.
# - V-measure: Based on homogeneity (do clusters contain mostly one class?) and
#   completeness (are all members of a class grouped together?), but it does not
#   correct for chance. If you run a random clustering, V-measure might still
#   give a misleadingly non-zero score, unlike AMI or ARI.
