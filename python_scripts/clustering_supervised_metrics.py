# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Clustering performance metrics in the presence of labels
#
# In this notebook we briefly introduce how to deal with text data. Then we
# introduce the use of supervised metrics to evaluate a clustering model.
#
# Previously, we saw how categorical features can be converted into numbers
# using techniques like one-hot encoding, where each category is assigned a
# unique position in a vector. We can apply a similar idea to text: treat each
# unique word as a feature (a column), and represent each document as a vector
# (a row) indicating the word counts in it. This encoding process is known as
# "vectorization". Here is a quick example using `CountVectorizer` to turn a few
# short phrases into a numerical table:

# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "This is a simple phrase",
    "This phrase is shorter",
    "The previous phrase is shorter than the first phrase",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# %% [markdown]
# Observe in particular that the words "the" and "phrase" are counted twice in
# the last document (the third row in the dataframe). Readers interested can
# visit the [scikit-learn example on comparing vectorization
# strategies](https://scikit-learn.org/stable/auto_examples/text/plot_hashing_vs_dict_vectorizer.html).
#
# Now let's use the BBC News dataset to show how text documents can be cluster
# by topic. The dataset consists of 1,250 samples divided into 5 different
# categories: "business", "entertainment", "sport", "politics" and "tech".

# %%
data = pd.read_csv("../datasets/bbc_news.csv")
data

# %% [markdown]
# We start by trying a model consisting of `StringEncoder`, which vectorizes the
# text while keeping the feature space reasonably small, followed by `KMeans`.
#
# In a previous notebook we increased the default value of `n_init` as in that
# case data turned out to be unevenly distributed. In this case we also set
# `n_init=5` but for a different reason. When working with high-dimensional data
# (i.e. data with a large number of features) such as vectorized text, k-means
# can initialize centroids on extremely isolated data points that can stay their
# own centroids all along. You can see the [example on clustering sparse
# data](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering-sparse-data-with-k-means)
# for more information.

# %%
from skrub import StringEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

model = make_pipeline(StringEncoder(), KMeans(n_init=5, random_state=0))
model

# %% [markdown]
# Even though clustering is an unsupervised learning method, we have access to
# labels for the categories that were assigned by the BBC editors, allowing us
# to evaluate how well the clusters found using our models match these labels.
# While we could turn this into a multiclass classification problem and use
# metrics such as the accuracy, this approach has limitations. For example, an
# article on "tech for entertainment" might be assigned to a cluster that
# doesn't perfectly match its original label, even though it shares similarities
# with both "tech" and "entertainment" articles. This is why we focus on metrics
# that evaluate clustering beyond strict label matching.
#
# In this notebook, we'll use two metrics: V-measure and Adjusted Rand Index
# (ARI). The V-measure addresses overlaps by measuring both homogeneity (how
# pure the clusters are) and completeness (how well each category is grouped
# together). For instance, if an article on "tech and entertainment" is placed
# in a cluster mostly about "entertainment," this can still be a reasonable
# result. V-measure ranges from 0 to 1, where 1 indicates perfect clustering
# (both pure and complete), and 0 means the clustering is ineffective.
#
# The Adjusted Rand Index (ARI) measures the similarity between the predicted
# clusters and human-assigned labels, adjusting for random labeling. For the BBC
# News dataset, it compares pairs of articles to see if they are in the same
# cluster in both the predicted and human-assigned labels. High ARI means that
# articles from the same category are grouped together, and articles from
# different categories are separated. ARI ranges from -1 (worse than random
# clustering) to 1 (perfect clustering), with 0 indicating a model that assigns
# cluster labels at random. A high ARI value shows good alignment between the
# predicted clusters and the human-assigned labels, while a low ARI suggests
# poor clustering performance.

# %%
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, train_size=0.9, random_state=0)

n_clusters_values = range(2, 11)
scoring_names = {
    "V-measure": "v_measure_score",
    "ARI": "adjusted_rand_score",
}
fig, ax = plt.subplots(figsize=(6, 4))
for scoring in scoring_names.values():
    ValidationCurveDisplay.from_estimator(
        model,
        data["text"],
        data["category"],
        param_name="kmeans__n_clusters",
        param_range=n_clusters_values,
        score_type="train",
        scoring=scoring,
        std_display_style="errorbar",
        cv=cv,
        n_jobs=4,
        ax=ax,
    )
ax.set(
    ylim=(-0.1, 1.1),
    xlabel="Number of components",
    ylabel="Score",
    title="Validation curves for\nStringEncoder + KMeans",
)
handles, _ = ax.get_legend_handles_labels()
_ = ax.legend(handles=handles, labels=scoring_names.keys())

# %% [markdown]
# We observe that both V-measure and ARI reach their maximum when
# `n_clusters=5`, which matches the number of human-assigned categories in the
# dataset. This alignment reflects both good editorial labels (as the categories
# are well-defined and internally consistent) as well as a clustering pipeline
# that can reasonably extract the structure in the data that matches the human
# intuition.
#
# But the question may arise, if we didn't have access to labels at all, would
# the silhouette score also lead us to chose `n_clusters=5`?

# %%
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

n_clusters_values = range(2, 11)
for random_state in range(1, 6):
    data_subsample, _ = train_test_split(
        data["text"], train_size=0.9, random_state=random_state
    )
    scores = []
    for n_clusters in n_clusters_values:
        model[-1].set_params(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(data_subsample)
        data_transformed = model[:-1].transform(data_subsample)
        score = silhouette_score(data_transformed, cluster_labels)
        scores.append(score)

    plt.plot(n_clusters_values, scores, color="tab:blue", alpha=0.2)
    plt.xlabel("Number of clusters (n_clusters)")
    plt.ylabel("Silhouette score")
    _ = plt.title("Silhouette score for varying n_clusters", y=1.01)

# %% [markdown]
# The fact that the silhouette score favors larger values for `n_clusters` than
# the supervised metrics we saw before suggests that, in terms of cluster
# tightness and separation, having more clusters leads to better-defined
# clusters. This could be a case where the data contains subclusters or finer
# distinctions within the categories that the algorithm can capture with more
# clusters. Essentially, while there may be 5 high-level categories, the
# subcategories within those 5 groups (such as articles that are very niche
# within "tech" or "politics") could benefit from additional clusters.
#
# To reduce the variability across resamplings, we can normalize the results
# coming from the `StringEncoder`. Here `Normalizer` scales each sample
# individually to have unit length. This ensures that clustering is driven by
# the relative angle of the features coming from the vectorizer, rather than the
# overall size of the vector. If two documents use the same set of dominant
# words, in similar proportions, their vectors end up pointing in roughly the
# same direction, resulting in a small relative angle.

# %%
from sklearn.preprocessing import Normalizer

model = make_pipeline(
    StringEncoder(), Normalizer(copy=False), KMeans(n_init=5, random_state=0)
)

for random_state in range(1, 6):
    data_subsample, _ = train_test_split(
        data["text"], train_size=0.9, random_state=random_state
    )
    scores = []
    for n_clusters in n_clusters_values:
        model[-1].set_params(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(data_subsample)
        data_transformed = model[:-1].transform(data_subsample)
        score = silhouette_score(data_transformed, cluster_labels)
        scores.append(score)

    plt.plot(n_clusters_values, scores, color="tab:blue", alpha=0.2)
    plt.xlabel("Number of clusters (n_clusters)")
    plt.ylabel("Silhouette score")
    _ = plt.title("Silhouette score for varying n_clusters", y=1.01)

# %% [markdown]
# That didn't work as intended.
