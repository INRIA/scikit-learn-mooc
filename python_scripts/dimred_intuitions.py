# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Geometric Intuitions of PCA
#
# Principal Component Analysis (PCA) is a dimensionality reduction technique. It
# is an unsupervised learning method, i.e. it works with features only but there
# is **no target variable**.
#
# The objective of this notebook is to build up our geometric intuition using a
# simple 2D feature space. For this purpose we load the penguins dataset and
# keep two features that are correlated: the length and depth of the culmen. As
# this is an unsupervised task, the idea is not to predict either feature, for
# example by fitting a regression line, but rather to measure how much
# independent information can be obtained from each of them.

# %%
import pandas as pd
import matplotlib.pyplot as plt

penguins = pd.read_csv("../datasets/penguins_classification.csv")
penguins = penguins[penguins["Species"] == "Chinstrap"]
penguins = penguins.drop(columns="Species")
penguins.plot.scatter(x="Culmen Length (mm)", y="Culmen Depth (mm)")

# %% [markdown]
# ## Finding the principal component
#
# PCA is intended to find new features (called principal components, or "PC")
# that capture enough of the structure in our data.
#
# The first PC is the direction along which our data varies the most. Because
# our features are correlated, there is a clear dominant pattern in the data
# which PCA can identify. The second PC would align with the direction with
# second to most variance, and so on.
#
# In our case we begin by extracting both components to understand the full
# picture. **This is not dimensional reduction yet**, these components are
# linear combinations of the original features, in other words, just a rotation
# to more convenient axes.

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(penguins)

# %% [markdown]
# ### Understanding PCA attributes
#
# After fitting, `PCA` provides a `components_` attribute, which is an array of
# shape `(n_components, n_features)`. Each row is a PC, each column corresponds
# to original features:

# %%
pca.components_

# %% [markdown]
# These numbers tell us how to create the new features:

# %%
feature_names = penguins.columns.tolist()
for i, component in enumerate(pca.components_):
    terms = " + ".join(
        f"{w:.1f} * {f}" for w, f in zip(component, feature_names)
    )
    print(f"PC{i + 1} = {terms}")

# %% [markdown]
# The components are perpendicular (orthogonal) to each other. Indeed,
# components in the space of reduced dimensions work as new coordinate axes. We
# can plot them to better visualize the effect.

# %%
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

penguins.plot.scatter(
    x="Culmen Length (mm)", y="Culmen Depth (mm)", label="Original data", ax=ax
)
center = penguins[["Culmen Length (mm)", "Culmen Depth (mm)"]].mean().values
for component, color, label in zip(
    pca.components_,
    ["red", "blue"],
    ["First PC", "Second PC"],
):
    # Draw axes defining the PC space
    endpoints = np.array([center - component, center + component])
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color=color,
        linewidth=2,
        label=label,
        alpha=0.8,
    )

ax.set_title("Principal Components as New Feature Directions")
ax.legend()
ax.axis("equal")
plt.show()

# %% [markdown]
# The red line shows the first PC. It follows the correlation pattern in our
# data. The blue line (second PC) is perpendicular and captures the remaining
# variance.
#
# Another important attribute of `PCA` is the `explained_variance_`. By plotting
# it we can confirm quantitatively what we know: the first PC "explains" most of
# the variance. In other words, our data is more spread over the direction of
# the first PC.

# %%
fig, ax = plt.subplots()
bars = ax.barh(
    range(1, len(pca.explained_variance_) + 1),
    pca.explained_variance_.round(decimals=1),
)
ax.bar_label(bars)
ax.set_xlim([0, 14])
ax.set_yticks([1, 2], labels=["PC1", "PC2"])
ax.set_xlabel("eigenvalues")
ax.set_ylabel("PCA features")
ax.set_title("Variance Explained by Principal Components (PCA)", y=1.05)
plt.show()

# %% [markdown]
# The `explained_variance_`, is the statistical variance (as computed by the
# method `var`) of the PC space, in other words, in the new space obtained by
# transforming the original `penguins.values` using the matrix defined by
# `pca.components_`.

# %%
print(pca.explained_variance_)
print(
    (penguins.values @ pca.components_.T).var(axis=0, ddof=1)
)  # ddof scales by n_samples-1

# %% [markdown]
# Since the variance is the square of the standard deviation, we have to take
# the square root to recover the original scale in millimeters:

# %%
pca_std = np.sqrt(pca.explained_variance_)
print(f"Std along the first PC : {pca_std[0]:.3f} mm")
print(f"Std along the second PC : {pca_std[1]:.3f} mm")

# %% [markdown]
# If we are more interested in the proportion of the total variance carried by
# each component, and not so much on the original scale, we can make use of the
# `explained_variance_ratio_` attribute:

# %%
# total_explained_variance = pca.explained_variance_.sum()
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i + 1} carries {100 * var_ratio:.1f}% of the total variance")

# %% [markdown]
# Percentages can also be obtained directly from the `explained_variance_`:

# %%
100 * pca.explained_variance_ / pca.explained_variance_.sum()

# %% [markdown]
# Notice that how much data spreads over a given direction strongly depends on
# the scale of the original features, but we will discuss the need for scaling
# in the next notebook.

# %% [markdown]
# ## Dimensionality reduction from 2D to 1D
#
# As we saw, PCA is a transformation to a PC space where axes align with the
# directions of maximum variance. What makes this interesting is when used as a
# dimensionality reduction technique. In this case we can represent our 2D data
# using just the first principal component, effectively reducing from 2 features
# to 1.
#
# This works well here because our original features were correlated. One degree
# of freedom suffices to capture most of the information, which is the overall
# size of the penguin.

# %%
# Transform to principal component space
pca_1d = PCA(n_components=1)
penguins_transformed = pca_1d.fit_transform(penguins)

print(f"Original shape: {penguins.shape} (samples, features)")
print(f"Transformed shape: {penguins_transformed.shape} (samples, components)")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

penguins.plot.scatter(
    x="Culmen Length (mm)",
    y="Culmen Depth (mm)",
    label="Original data",
    alpha=0.6,
    ax=ax1,
)
ax1.set_title("Original 2D feature space")
ax1.axis("equal")

ax2.scatter(
    penguins_transformed.ravel(),
    np.zeros(len(penguins_transformed)),
    alpha=0.6,
)
ax2.set_xlabel("First Principal Component")
ax2.set_title(
    f"Reduced 1D space ({pca_1d.explained_variance_ratio_[0]:.0%} variance retained)"
)

# %% [markdown]
# The transformation creates a new 1D representation where samples that were
# close in the original 2D space remain close in the new 1D space. The structure
# is preserved.

# %% [markdown]
# ## Loss of information during reconstruction
#
# When we use fewer components than original features, we lose some information.
# The `inverse_transform` method shows us what our data looks like when reconstructed
# from the reduced representation.

# %%
penguins_reconstructed = pca_1d.inverse_transform(penguins_transformed)

fig, ax = plt.subplots(figsize=(7, 5))
penguins.plot.scatter(
    x="Culmen Length (mm)",
    y="Culmen Depth (mm)",
    label="Original data",
    alpha=0.6,
    ax=ax,
)
ax.scatter(
    penguins_reconstructed[:, 0],
    penguins_reconstructed[:, 1],
    alpha=0.6,
    s=30,
    color="red",
    label="Reconstruction",
)
ax.axis("equal")
ax.legend()
ax.set_title("Original vs reconstructed feature space")
plt.show()

# %% [markdown]
# The reconstructed points all lie on a line, that is, we have lost the variance
# perpendicular to it, but retained the main pattern. The `inverse_transform` is
# a rotation back to the original axes, that in this case maps the 1D
# representation back into 2D. The variance along the remaining component was
# already discarded during the forward projection, which is why the points
# collapse onto a line.
#
# One geometrically intuitive way to quantify the information lost during
# dimensionality reduction is the squared Euclidean distance between the
# original feature vector and its reconstruction, then averaged over all
# samples. This is different from a flat mean over all elements, as we first sum
# the squared differences across features (`axis=1`), preserving the geometric
# notion of distance in more than 1 dimension, and only then averaging over
# samples.

# %%
reconstruction_error = np.mean(
    np.sum((penguins - penguins_reconstructed) ** 2, axis=1)
)
print(f"Mean squared reconstruction error: {reconstruction_error:.4f}")

# %% [markdown]
# ## PCA vs Linear Regression
#
# From this example it might be tempting to compare PCA with linear regression
# since both can produce lines through data. To illustrate the difference, let's
# pretend for a moment that "Culmen Depth (mm)" is a target for regression.

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(penguins[["Culmen Length (mm)"]], penguins["Culmen Depth (mm)"])

x1_range = pd.DataFrame(
    {
        "Culmen Length (mm)": np.linspace(
            penguins["Culmen Length (mm)"].min(),
            penguins["Culmen Length (mm)"].max(),
            100,
        )
    }
)
x2_pred = lr.predict(x1_range)
center = pca_1d.mean_
direction = pca_1d.components_[0]
t = np.linspace(-8, 9, 100)
pc_line = center + t[:, np.newaxis] * direction

fig, ax = plt.subplots(figsize=(8, 6))
penguins.plot.scatter(
    x="Culmen Length (mm)",
    y="Culmen Depth (mm)",
    label="Original data",
    alpha=0.6,
    ax=ax,
)
ax.plot(x1_range, x2_pred, "b-", linewidth=2, label="Regression line")

ax.plot(pc_line[:, 0], pc_line[:, 1], "r-", linewidth=2, label="First PC")

ax.set_title("Regression line vs First PC")
ax.legend()
plt.show()

# %% [markdown]
# The slopes are slightly different. Indeed :
# - Linear regression minimises the vertical distance (residuals in the
#   y-direction only) between each point and the line. It treats the two
#   features asymmetrically, with one as predictor and one as target.
# - PCA minimises the perpendicular distance from each point to the line. It
#   treats both features symmetrically, with no notion of predictor/target.
#
# ## Key Takeaways
#
# - PCA is **an unsupervised learning method**. It works with features only,
#   treating them symmetrically.
# - PCA creates a new feature space defined by the principal components as
#   weighted combinations of the original features.
# - If the principal components space has lower dimension that the original
#   feature space, PCA does dimensionality reduction.
# - By keeping components with high variance, we preserve the main patterns in
#   the data.
# - Using fewer components than original dimensions means accepting some
#   information loss for the benefit of simplicity.
#
# Next, we'll explore how PCA behaves with outliers and noise, helping you
# understand when PCA works well and when to be cautious.
