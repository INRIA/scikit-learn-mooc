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
# introduce the use of performance metrics to evaluate a clustering model when
# we have access to labeled data. The reason is that clustering gives us a way
# to explore whether the structure in the data supports the labels we've been
# given, or even to question how well-defined those labels are in the first
# place.
#
# ## Feature engineering for text data
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
X_vectorized = vectorizer.fit_transform(docs)

pd.DataFrame(
    X_vectorized.toarray(), columns=vectorizer.get_feature_names_out()
)

# %% [markdown]
# Observe in particular that the words "the" and "phrase" are counted twice in
# the last document (the third row in the dataframe). This preprocessor creates
# as many features as unique words ocurring in the data, therefore the dimension
# of the feature space can become very large.
#
# Readers interested can visit the [scikit-learn example on comparing
# vectorization
# strategies](https://scikit-learn.org/stable/auto_examples/text/plot_hashing_vs_dict_vectorizer.html)
# for more information.
#
# Now let us use the BBC News dataset to show how text documents can be cluster
# by topic.

# %%
data = pd.read_csv("../datasets/bbc_news.csv")
data

# %% [markdown]
# The dataset consists of 1,250 samples divided into 5 different categories:
# "business", "entertainment", "sport", "politics" and "tech".

# %%
data["category"].value_counts()

# %% [markdown]
# We start by preprocessing the text data using `StringEncoder`, which is an
# alternative to `CountVectorizer` that encodes text while keeping the dimension
# of the feature space reasonably small, even if the number of unique words is
# very large. This encoder is well suited to cluster text using `KMeans`.

# %%
from skrub import StringEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

model = make_pipeline(StringEncoder(), KMeans(n_clusters=3, random_state=0))
cluster_labels = model.fit_predict(data["text"])
pd.Series(cluster_labels).value_counts()

# %% [markdown]
# Our pipeline has grouped the documents into 3 clusters, even though the
# dataset contains 5 categories assigned by BBC editors. We chose this on
# purpose, to show that k-means is able to cluster data even if
# those clusters do not match the predefined categories.
#
# ## Supervised metrics for clustering evaluation
#
# Even though clustering is an unsupervised learning method, we have access to
# labels for the categories that were assigned by the BBC editors, allowing us
# to evaluate how well the clusters found using our models match those labels.
#
# We could try to use classification metrics such as the accuracy. However,
# clustering labels are arbitrary and their number does not need to match the
# number of predefined categories (we just saw in the example above). More
# importantly, we don't assume a predefined mapping between cluster labels and
# editorial categories, and we don't need one to quantify their agreement. This
# is where supervised clustering metrics come in.
#
# In this notebook, we'll use two metrics: V-measure and Adjusted Rand Index
# (ARI). The V-measure addresses overlaps by evaluating two properties:
# - homogeneity: each cluster contains only members of a single class;
# - completeness: all members of a given class are assigned to the same cluster.
#
# V-measure ranges from 0 to 1, where 1 indicates perfect match between the
# clustering labels and the human labels, both in terms of homogeneity and
# completeness.
#
# The Adjusted Rand Index (ARI) also measures the similarity between the
# predicted clusters and human-assigned labels, adjusting for random labeling.
# For the BBC News dataset, it compares pairs of articles to see whether if they
# are in the same cluster in both the predicted and human-assigned labels. High
# ARI means that articles from the same category are grouped together, and
# articles from different categories are separated. ARI ranges from -1 (worse
# than random clustering) to 1 (perfect clustering), with 0 indicating a model
# that assigns cluster labels at random.
#
# In other words, both V-measure and ARI follow a "higher is better" convention.
#
# Read more in the User Guide for
# [V-measure](https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure)
# and the [Rand
# index](https://scikit-learn.org/stable/modules/clustering.html#rand-index).

# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, train_size=0.75, random_state=0)
data_encoded = StringEncoder().fit_transform(data["text"])
n_clusters_values = range(2, 11)
scoring_names = {
    "V-measure": "v_measure_score",
    "ARI": "adjusted_rand_score",
}
fig, ax = plt.subplots(figsize=(6, 4))
for scoring in scoring_names.values():
    ValidationCurveDisplay.from_estimator(
        KMeans(n_init=5, random_state=0),
        data_encoded,
        data["category"],
        param_name="n_clusters",
        param_range=n_clusters_values,
        score_type="train",
        scoring=scoring,
        std_display_style="errorbar",
        cv=cv,
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
# We observe that both V-measure and ARI are much better than chance. Even more
# they reach their maximum value when `n_clusters=5`. This is not surprising
# because this is the number of human-assigned categories in the dataset. The
# alignment reflects both good editorial labels (as the categories are
# well-defined and internally consistent) as well as a clustering pipeline that
# can reasonably extract the structure in the data that matches the human
# intuition.
#
# But the question may arise, if we didn't have access to labels at all, would
# the silhouette score also lead us to chose `n_clusters=5`?

# %%
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

n_clusters_values = range(2, 16)
for random_state in range(1, 11):
    data_subsample, _ = train_test_split(
        data_encoded, train_size=0.75, random_state=random_state
    )
    scores = []
    for n_clusters in n_clusters_values:
        model = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
        cluster_labels = model.fit_predict(data_subsample)
        score = silhouette_score(data_subsample, cluster_labels)
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
# within "tech" or "politics") could benefit from additional clusters. But this
# is probably not what the editors want in practice. Having too fine-grained
# topics would make the navigation on their website too confusing.
#
# Finally, notice that we used extra supervised information to quantitatively
# assess the quality of the match between the clusters found by k-means and our
# interpretation. In practice, this is often impossible, as we do not have
# access to human assigned labels for each row in the data. Or, if we have, we
# might want to use them to train the clustering model, but instead we would
# rather use them as the target variable to train a supervised classifier.
