# %% [markdown]
# # Importance of decision tree hyperparameters on generalization
#
# In this notebook, we will illustrate the importance of some key
# hyperparameters on the decision tree ; we will demonstrate their effects on
# the classification and regression problems we saw previously.
#
# First, we will load the classification and regression datasets.

# %%
import pandas as pd

data_clf_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_clf_column = "Species"
data_clf = pd.read_csv("../datasets/penguins_classification.csv")

# %%
data_reg_columns = ["Flipper Length (mm)"]
target_reg_column = "Body Mass (g)"
data_reg = pd.read_csv("../datasets/penguins_regression.csv")

# %% [markdown]
# ## Create helper functions
#
# We will create two functions that will:
#
# * fit a decision tree on some training data;
# * show the decision function of the model.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")


def plot_classification(model, X, y, ax=None):
    from sklearn.preprocessing import LabelEncoder
    model.fit(X, y)

    range_features = {
        feature_name: (X[feature_name].min() - 1, X[feature_name].max() + 1)
        for feature_name in X.columns
    }
    feature_names = list(range_features.keys())
    # create a grid to evaluate all possible samples
    plot_step = 0.02
    xx, yy = np.meshgrid(
        np.arange(*range_features[feature_names[0]], plot_step),
        np.arange(*range_features[feature_names[1]], plot_step),
    )

    # compute the associated prediction
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")
    if y.nunique() == 3:
        palette = ["tab:red", "tab:blue", "black"]
    else:
        palette = ["tab:red", "tab:blue"]
    sns.scatterplot(
        x=data_clf_columns[0], y=data_clf_columns[1], hue=target_clf_column,
        data=data_clf, ax=ax, palette=palette)

    return ax


# %%
def plot_regression(model, X, y, ax=None):
    model.fit(X, y)

    X_test = pd.DataFrame(
        np.arange(X.iloc[:, 0].min(), X.iloc[:, 0].max()),
        columns=X.columns,
    )
    y_pred = model.predict(X_test)

    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(x=X.iloc[:, 0], y=y, color="black", alpha=0.5, ax=ax)
    ax.plot(X_test, y_pred, linewidth=4)

    return ax


# %% [markdown]
# ## Effect of the `max_depth` parameter
#
# In decision trees, the most important parameter to get a trade-off between
# under-fitting and over-fitting is the `max_depth` parameter. Let's build
# a shallow tree and then deeper tree (for both classification and regression).


# %%
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

max_depth = 2
tree_clf = DecisionTreeClassifier(max_depth=max_depth)
tree_reg = DecisionTreeRegressor(max_depth=max_depth)

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
plot_classification(tree_clf, data_clf[data_clf_columns],
                    data_clf[target_clf_column], ax=axs[0])
plot_regression(tree_reg, data_reg[data_reg_columns],
                data_reg[target_reg_column], ax=axs[1])
fig.suptitle(f"Shallow tree with a max-depth of {max_depth}")
plt.subplots_adjust(wspace=0.3)


# %%
max_depth = 30
tree_clf = DecisionTreeClassifier(max_depth=max_depth)
tree_reg = DecisionTreeRegressor(max_depth=max_depth)

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
plot_classification(tree_clf, data_clf[data_clf_columns],
                    data_clf[target_clf_column], ax=axs[0])
plot_regression(tree_reg, data_reg[data_reg_columns],
                data_reg[target_reg_column], ax=axs[1])
fig.suptitle(f"Deep tree with a max-depth of {max_depth}")
plt.subplots_adjust(wspace=0.3)

# %% [markdown]
# For both classification and regression setting, we can observe that
# increasing the depth will make the tree model more expressive. However, a
# tree that is too deep will overfit the training data, creating partitions
# which are only correct for "outliers". The `max_depth` is one of the
# hyperparameters that one should optimize via cross-validation and
# grid-search.

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": np.arange(2, 10, 1)}
tree_clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
tree_reg = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
plot_classification(tree_clf, data_clf[data_clf_columns],
                    data_clf[target_clf_column], ax=axs[0])
plot_regression(tree_reg, data_reg[data_reg_columns],
                data_reg[target_reg_column], ax=axs[1])
axs[0].set_title(f"Optimal depth found via CV: "
                 f"{tree_clf.best_params_['max_depth']}")
axs[1].set_title(f"Optimal depth found via CV: "
                 f"{tree_reg.best_params_['max_depth']}")
plt.subplots_adjust(wspace=0.3)

# %% [markdown]
# ## Other hyperparameters in decision trees
#
# The `max_depth` parameter offers a possibility to have a global impact on
# the full tree structure. Indeed, by limiting the depth, you will stop all
# the branch trees to grow after this specific level.
#
# The other hyperparameters in decision trees provide a more fine grain
# optimization, acting at the level of the node and leaf. This is particularly
# interesting if the grown tree has an asymmetric shape where it could be
# beneficial to let the tree being more expressive in some ramifications.
#
# We will craft a toy dataset to illustrate this behavior. We will generate a
# dataset composed of 2 subsets: one subset where a clear separation will be
# found by the tree and a another subset where samples from both classes will
# be mixed. It implies that a decision tree will need more splits to classify
# properly samples from the second subset than from the first subset.

# %%
from sklearn.datasets import make_classification, make_blobs

data_clf_columns = ["Feature #0", "Feature #1"]
target_clf_column = "Class"

X_1, y_1 = make_classification(
    n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=1,
    n_redundant=0, class_sep=0.5, random_state=0)
X_2, y_2 = make_blobs(
    n_samples=300, centers=[[4, 6], [7, 0]], random_state=0)

X = np.concatenate([X_1, X_2], axis=0)
y = np.concatenate([y_1, y_2])
data_clf = np.concatenate([X, y[:, np.newaxis]], axis=1)
data_clf = pd.DataFrame(
    data_clf, columns=data_clf_columns + [target_clf_column])

# %% [markdown]
# We can visualize the dataset that we just generated.

# %%
sns.scatterplot(
    x=data_clf_columns[0], y=data_clf_columns[1], hue=target_clf_column,
    data=data_clf)

# %% [markdown]
# From our tree understanding, we should have the intuitions that the separated
# blobs should be easily separable in the first level of the tree.
#
# We will check if a `max_depth=3` will be enough to separate the distinct
# blobs.

# %%
_, ax = plt.subplots(figsize=(6, 6))
tree_clf = DecisionTreeClassifier(max_depth=3)
plot_classification(tree_clf, data_clf[data_clf_columns],
                    data_clf[target_clf_column], ax=ax)

# %% [markdown]
# Indeed, we see that red blob on the top and the blue blob on the right of
# the plot are perfectly separated. However, the tree is still making mistakes
# in the area where the blobs are mixed together. We can check the tree
# representation.

# %%
from sklearn.tree import plot_tree

_, ax = plt.subplots(figsize=(6, 6))
_ = plot_tree(tree_clf, ax=ax, feature_names=data_clf_columns)

# %% [markdown]
# After a depth of level 2, we see that the two blobs of 150 and 152 samples
# are respectively separated from the rest. Therefore, increasing the depth
# will just continue to split the imperfectly classified data.

 # %%
_, ax = plt.subplots(figsize=(6, 6))
tree_clf = DecisionTreeClassifier(max_depth=5)
plot_classification(tree_clf, data_clf[data_clf_columns],
                    data_clf[target_clf_column], ax=ax)

# %%
_, ax = plt.subplots(figsize=(11, 7))
_ = plot_tree(tree_clf, ax=ax, feature_names=data_clf_columns)

# %% [markdown]
# We see, that the tree continue to split the area where misclassification
# occurs. Thus, we have a tree structure which asymmetric with 2 leaves
# declared really early and the remaining branch which needs to grow.
#
# In this scenario, `max_depth` will limit the growth of all branches in the
# same manner, even if some ramifications would require more splits. The
# hyperparameters `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`,
# or `min_impurity_decrease` allows to grow asymmetric trees and apply a
# constraint at the leaves or nodes level. Let's see the effect of
# `min_samples_leaf`.

# %%
 # %%
_, ax = plt.subplots(figsize=(6, 6))
tree_clf = DecisionTreeClassifier(min_samples_leaf=60)
plot_classification(tree_clf, data_clf[data_clf_columns],
                    data_clf[target_clf_column], ax=ax)

# %%
_, ax = plt.subplots(figsize=(8, 8))
_ = plot_tree(tree_clf, ax=ax, feature_names=data_clf_columns)

# %% [markdown]
# This hyperparameter allows to have leave with a minimum number of samples.
# Therefore, we see that we can built different branches individually and the
# criterion to stop splitting will be linked to the leaf value instead of at
# the level of the tree.
