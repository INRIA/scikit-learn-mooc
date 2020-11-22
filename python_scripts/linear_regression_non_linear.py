# %% [markdown]
# # Beyond linear separation in classification
#
# As we saw in regression, the linear classification model expects the data
# to be linearly separable. When this assumption does not hold, the model
# is not expressive enough to properly fit the data. One needs to apply the
# same tricks as in regression: feature augmentation (potentially using
# expert-knowledge) or using a kernel based method.
#
# We will provide examples where we will use a kernel support vector machine
# to perform classification on some toy-datasets where it is impossible to
# find a perfect linear separation.
#
# First, we redefine our plotting utility to show the decision boundary of a
# classifier.

# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(fitted_classifier, range_features, ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    feature_names = list(range_features.keys())
    # create a grid to evaluate all possible samples
    plot_step = 0.02
    xx, yy = np.meshgrid(
        np.arange(*range_features[feature_names[0]], plot_step),
        np.arange(*range_features[feature_names[1]], plot_step),
    )

    # compute the associated prediction
    Z = fitted_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")

    return ax


# %% [markdown]
# We will generate some synthetic data with special pattern which are known to
# be non-linear.

# %%
import pandas as pd
from sklearn.datasets import (
    make_moons, make_classification, make_gaussian_quantiles,
)

X_moons, y_moons = make_moons(n_samples=500, noise=.13, random_state=42)
X_class, y_class = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    random_state=2,
)
X_gauss, y_gauss = make_gaussian_quantiles(
    n_samples=500, n_features=2, n_classes=2, random_state=42,
)

datasets = [
    [pd.DataFrame(X_moons, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_moons, name="class")],
    [pd.DataFrame(X_class, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_class, name="class")],
    [pd.DataFrame(X_gauss, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_gauss, name="class")],
]
range_features = {"Feature #0": (-5, 5), "Feature #1": (-5, 5)}

# %% [markdown]
# We will first visualize the different datasets.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

_, axs = plt.subplots(ncols=3, sharey=True, figsize=(14, 4))

for ax, (X, y) in zip(axs, datasets):
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y,
                    palette=["tab:red", "tab:blue"], ax=ax)

# %% [markdown]
# Inspecting these three datasets, it is clear that a linear model cannot
# separate the two classes. Now, we will train a SVC classifier where we will
# use a linear kernel to show the limitation of such linear model on the
# following dataset

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

linear_model = make_pipeline(StandardScaler(), SVC(kernel="linear"))

_, axs = plt.subplots(ncols=3, sharey=True, figsize=(14, 4))
for ax, (X, y) in zip(axs, datasets):
    linear_model.fit(X, y)
    plot_decision_function(linear_model, range_features, ax=ax)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y,
                    palette=["tab:red", "tab:blue"], ax=ax)
    ax.set_title(f"Accuracy: {linear_model.score(X, y):.3f}")

# %% [markdown]
# As expected, the linear model parametrization is not enough to adapt the
# synthetic dataset.
#
# Now, we will fit an SVC with an RBF kernel that will handle the non-linearity
# using the kernel trick.

# %%
kernel_model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

_, axs = plt.subplots(ncols=3, sharey=True, figsize=(14, 4))
for ax, (X, y) in zip(axs, datasets):
    kernel_model.fit(X, y)
    plot_decision_function(kernel_model, range_features, ax=ax)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y,
                    palette=["tab:red", "tab:blue"], ax=ax)
    ax.set_title(f"Accuracy: {kernel_model.score(X, y):.3f}")

# %% [markdown]
# In this later case, we can see that the accuracy is close to be perfect and
# that the decision boundary is non-linear. Thus, kernel trick or data
# augmentation are the tricks to make a linear classifier more expressive.

# %% [markdown]
# # Main take away
#
# - a linear model as a specific parametrization defined by some weights and an
#   intercept;
# - linear models require to scale the data before to be trained;
# - regularization allows to fight over-fitting;
# - the regularization parameter needs to be fine tuned for each application;
# - linear models can be used with data presenting non-linear links but require
#   extra work such as the use of data augmentation or kernel trick.
