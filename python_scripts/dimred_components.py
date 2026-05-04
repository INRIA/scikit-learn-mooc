# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Choice of `n_components` in the unsupervised case
#
# In the previous exercise we treated `n_components` as a hyperparameter in a
# **supervised** pipeline. Unlike typical hyperparameters, the evaluation metric
# plateaus after a certain number of components, beyond which adding more only
# increases fit time. When a ground truth label is available, such plateau gives
# a natural stopping criterion. When it is not, you need indirect criteria. Each
# criterion encodes a different assumption about what "enough components" means,
# and none of them is universally correct.
#
# In this notebook we work through the main criteria for choosing
# `n_components` and their implications:
#
# - Reading cumulative explained variance curves and applying 90%/95%
#   thresholds;
# - using the Kaiser criterion as a threshold-free heuristic;
# - checking how stable both choices are under resampling;
# - using silhouette score to evaluate a clustering pipeline, comparing what the
#   two criteria recommend.
#
# We use the Wine recognition dataset throughout, pretending we do not have
# access to the true cultivar labels.

# %%
from sklearn.datasets import load_wine

X, _ = load_wine(return_X_y=True, as_frame=True)

# %% [markdown]
# The dataset contains 178 wine samples from three cultivars grown in Italy.
# Each sample has 13 chemical measurements, such as alcohol content, acidity,
# and various phenolic compounds. The goal in the original task is to identify
# the cultivar from the chemistry alone.
#
# We use it here purely to practice selecting the number of PCA components on
# real data where the features have very different scales and units.

# %%
from skrub import TableReport

TableReport(X)

# %% [markdown]
# The dataset is composed of numerical features only, spanning different ranges
# of values, for instance, `proline` goes up to ~1680, while
# `nonflavanoid_phenols` stays below 0.7. By looking at the `Distributions` tab
# of the `TableReport` we see that the distribution do not have outliers, which
# means we can simply use `StandardScaler` to scale the data before reducing the
# dimensionality.
#
# ## Explained variance across all components
#
# We start with two standard ways of visualizing `explained_variance_ratio_`
# across components. They carry the same information, but they will help us
# define different selection criteria. For such purpose, we fit PCA with its
# default `n_components`, so all components are computed and no dimensionality
# reduction is applied yet.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), PCA())
pipe.fit(X)

pca = pipe.named_steps["pca"]
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
components = np.arange(1, len(explained) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(components, explained)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained variance ratio")
axes[0].set_title("Variance ratio per component\n(scree plot)")

axes[1].plot(components, cumulative, marker="o")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative explained variance")
_ = axes[1].set_title("Cumulative explained variance ratio")

# %% [markdown]
# The bar chart on the left is called a **scree plot**. It shows how much new
# variance each additional component brings. The cumulative curve on the right
# shows the total variance captured as you include more components.
#
# Historically, practitioners looked for an "elbow" in the scree plot, that is,
# the point where the bars stop dropping steeply and start to level off, similar
# to the elbow method in clustering. We will not dwell on it here because it is
# harder to interpret and less stable than the criteria we present below, but it
# is worth knowing the term.

# %% [markdown]
# ## Variance thresholds: 90% and 95%
#
# A common rule of thumb is to keep enough components to explain 90% or 95%
# of the total variance. `np.searchsorted` finds the first index where the
# cumulative variance crosses the threshold.

# %%
threshold_90 = np.searchsorted(cumulative, 0.90) + 1
threshold_95 = np.searchsorted(cumulative, 0.95) + 1

print(f"Components to reach 90% variance: {threshold_90}")
print(f"Components to reach 95% variance: {threshold_95}")

# %% [markdown]
# You can also let scikit-learn handle this directly by passing a float between
# 0 and 1 to the `n_components` parameter of `PCA`.

# %%
pipe_90 = make_pipeline(StandardScaler(), PCA(n_components=0.90))
pipe_90.fit(X)
print(
    f"n_components_ for 90% threshold: {pipe_90.named_steps['pca'].n_components_}"
)

# %% [markdown]
# Both thresholds are computed on the full dataset. But a PCA fitted on a
# slightly different sample would produce slightly different components and
# slightly different explained variance values. This raises a practical
# question: if you deploy a model using `threshold_90` components and claim
# it retains 90% of the variance, how much variance does it actually retain
# on a different sample of the same data?
#
# We simulate this by fitting PCA on 20 random 50% subsamples and overlaying
# the resulting cumulative variance curves.

# %%
from sklearn.model_selection import train_test_split

n_splits = 20
split_explained = []

for random_state in range(n_splits):
    X_split, _ = train_test_split(X, train_size=0.5, random_state=random_state)
    pipe_split = make_pipeline(StandardScaler(), PCA())
    pipe_split.fit(X_split)
    split_explained.append(
        pipe_split.named_steps["pca"].explained_variance_ratio_
    )

# %%
fig, ax = plt.subplots(figsize=(8, 4))

for ev in split_explained:
    ax.plot(
        np.arange(1, len(ev) + 1), np.cumsum(ev), color="tab:blue", alpha=0.2
    )

ax.plot(
    components, cumulative, color="tab:blue", linewidth=2, label="Full dataset"
)
ax.axhline(0.90, color="tab:orange", linestyle="--", label="90%")
ax.axhline(0.95, color="tab:red", linestyle="--", label="95%")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("Cumulative variance is not identical across subsamples")
_ = ax.legend()

# %% [markdown]
# Each faint line is one subsample. The spread is most visible near the
# threshold lines, which means a fixed number of components does not guarantee
# exactly 90% or 95% variance retained. We can quantify this directly.

# %%
split_cumulative = np.array([np.cumsum(ev) for ev in split_explained])

for n_comp in [threshold_90, threshold_95]:
    values = split_cumulative[:, n_comp - 1] * 100
    print(
        f"{n_comp} components → {values.mean():.1f} ± {values.std():.1f}% "
        f"variance retained across splits"
    )

# %% [markdown]
# This is what you should report when justifying a component choice in a
# deployed model. Rather than stating "we retain 90% of the variance", you can
# say "with N components, we retain X ± Y% of the variance across subsamples",
# which is a more honest and informative claim.
#
# ## The Kaiser criterion
#
# In the original Kaiser criterion (designed for factor analysis), you discard
# components whose eigenvalue is below 1, meaning they explain less variance
# than a single original variable would. The analog in PCA replaces the
# raw eigenvalue threshold with its equivalent in variance ratio terms: keep
# components that each explain more than 1/`n_features` of the total variance.

# %%
n_features = X.shape[1]
kaiser_threshold = 1 / n_features

print(f"Number of features: {n_features}")
print(f"Kaiser threshold (1/n_features): {kaiser_threshold:.4f}")
print(f"Components above threshold: {np.sum(explained > kaiser_threshold)}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(components, explained, label="Explained variance ratio")
ax.axhline(
    kaiser_threshold,
    color="tab:red",
    linestyle="--",
    label=f"Kaiser threshold (1/{n_features} ≈ {kaiser_threshold:.2f})",
)
ax.set_xlabel("Component")
ax.set_ylabel("Explained variance ratio")
ax.set_title(f"Kaiser criterion: keep components above 1/{n_features}")
_ = ax.legend()

# %% [markdown]
# The Kaiser criterion does not require choosing a target variance level, which
# makes it easier to apply. One natural question is whether it also leads to
# more stable recommendations than the threshold approach. We overlay the same
# subsamples on the scree plot to find out.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

for ev in split_explained:
    ax.plot(np.arange(1, len(ev) + 1), ev, color="tab:blue", alpha=0.2)

ax.plot(
    components, explained, color="tab:blue", linewidth=2, label="Full dataset"
)
ax.axhline(
    kaiser_threshold,
    color="tab:red",
    linestyle="--",
    label=f"Kaiser threshold (1/{n_features})",
)
ax.set_xlabel("Component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("Kaiser criterion stability across subsamples")
_ = ax.legend()

# %% [markdown]
# Notice that this curve reinforces the argument that finding an elbow depends
# heavily on the resampling and is not stable to resamplings.
#
# Observe also that the curve has sampling variability near the 1/`n_features`
# boundary. Because of that, this criterion does not lead to a unique choice for
# the number of components for this dataset.

# %%
import pandas as pd

kaiser_n = np.sum(explained > kaiser_threshold)
kaiser_counts = [np.sum(ev > kaiser_threshold) for ev in split_explained]
ax = pd.Series(kaiser_counts).value_counts().sort_index().plot.bar(rot=0)
_ = ax.set(
    title="Selected number of components across splits",
    xlabel="Number of components",
    ylabel="Counts",
)

# %% [markdown]
# The Kaiser criterion suggests 3 or 4 components at almost the same rate.
# Either choice is possible, but for the sake of this notebook, we retain
# the latest value of `kaiser_n` and find how much of the total explained
# variance it retains.

# %%
values = split_cumulative[:, kaiser_n - 1] * 100
print(
    f"{kaiser_n} components → {values.mean():.1f} ± {values.std():.1f}% "
    f"variance retained across splits"
)

# %% [markdown]
# ## Downstream check: does the component choice affect clustering?
#
# We now use KMeans as a downstream task to see whether the choice of
# `n_components` has any practical impact. We fit the same range of cluster
# counts under two pipelines: one using the Kaiser recommendation
# (`n_components=3`) and one using the 90% threshold (`n_components=8`).
#
# As in the clustering notebook, we plot the silhouette score against number of
# clusters and look for an elbow. Notice that silhouette is not comparable
# *across* different numbers of components, since the score depends on the space
# in which distances are computed, even if the score is normalized. We are not
# comparing the two curves against each other, instead we are checking whether
# each curve, read on its own, points to the same number of clusters.
#
# We use the silhouette score and plot one curve per subsample, similarly to
# what we did in the clustering chapter of the Associate course. The two axes
# share the same y-axis range since the silhouette score has the same scale in
# both panels, which makes the comparison honest.

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_clusters_range = range(2, 11)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, n_components, label in zip(
    axes,
    [kaiser_n, threshold_90],
    [
        f"Kaiser ({kaiser_n} components)",
        f"90% threshold ({threshold_90} components)",
    ],
):
    pipe_km = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        KMeans(random_state=0),
    )
    for random_state in range(1, 11):
        X_sub, _ = train_test_split(
            X, train_size=0.5, random_state=random_state
        )
        scores = []
        for k in n_clusters_range:
            pipe_km[-1].set_params(n_clusters=k)
            labels = pipe_km.fit_predict(X_sub)
            X_transformed = pipe_km[:-1].transform(X_sub)
            scores.append(silhouette_score(X_transformed, labels))
        ax.plot(n_clusters_range, scores, color="tab:blue", alpha=0.2)

    ax.set_xlabel("Number of clusters (n_clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(label)

# %% [markdown]
# The two panels tell a different story. With the Kaiser components, the peak at
# 3 clusters is clear and consistent across all subsamples. With 8 components
# the curves are much more spread out and there is no clear consensus.
#
# Keeping 92% variance is still capturing noise and therefore leading to high
# variance. Even in an unsupervised setting, we can conclude that the PCA step
# with 8 components is overfitting!
#
# Notice also that the silhouette scores are also systematically lower in the
# right panel. This is not because the quality of the clusters is worse in an
# absolute sense, but because of the curse of dimensionality. In
# high-dimensional spaces, all pairwise distances tend to concentrate around the
# same value (a phenomenon known as distance concentration). As a result, the
# within-cluster distance `a` and the nearest-cluster distance `b` become
# increasingly similar, so the numerator `b - a` shrinks relative to `max(a, b)`
# and the silhouette score is driven toward zero. For this reason we should not
# compare the absolute level of the scores between the two panels. What matters
# is the shape of each curve and where it peaks, not the y-axis value itself.
#
# ## Key Takeaways
#
# The Kaiser criterion trades one arbitrary choice (the 90 - 95% level) for
# another arbitrary choice (the 1/`n_features` rule). A priori neither is more
# stable under resampling, but both allow us to make informed choices even in a
# fully unsupervised pipeline.
#
# In the case of supervised pipelines, prefer cross-validation to tune the
# number of components as you would do for any other hyperparameter.
