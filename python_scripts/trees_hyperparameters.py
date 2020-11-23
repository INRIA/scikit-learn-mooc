# %% [markdown]
# # Importance of decision tree hyper-parameters on generalization
#
# In this notebook will illustrate the importance of some key hyper-parameters
# of the decision tree. We will illustrate it on both the classification and
# regression probelms that we previously used.
#
# ## Load the classification and regression datasets
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
    sns.scatterplot(
        x=data_clf_columns[0], y=data_clf_columns[1], hue=target_clf_column,
        data=data_clf, ax=axs[0], palette=["tab:red", "tab:blue", "black"])

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
# ### Effect of the `max_depth` parameter
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
plot_classification(
    tree_clf, data_clf[data_clf_columns], data_clf[target_clf_column],
    ax=axs[0]
)
plot_regression(
    tree_reg, data_reg[data_reg_columns], data_reg[target_reg_column],
    ax=axs[1])
fig.suptitle(f"Shallow tree with a max-depth of {max_depth}")
plt.subplots_adjust(wspace=0.3)


# %%
max_depth = 30
tree_clf = DecisionTreeClassifier(max_depth=max_depth)
tree_reg = DecisionTreeRegressor(max_depth=max_depth)

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
plot_classification(
    tree_clf, data_clf[data_clf_columns], data_clf[target_clf_column],
    ax=axs[0]
)
plot_regression(
    tree_reg, data_reg[data_reg_columns], data_reg[target_reg_column],
    ax=axs[1])
fig.suptitle(f"Deep tree with a max-depth of {max_depth}")
plt.subplots_adjust(wspace=0.3)

# %% [markdown]
# For both classification and regression setting, we can observe that
# increasing the depth will make the tree model more expressive. However, a
# tree that is too deep will overfit the training data, creating partitions
# which are only be correct for "outliers". The `max_depth` is one of the
# hyper-parameters that one should optimize via cross-validation and
# grid-search.

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": np.arange(2, 10, 1)}
tree_clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
tree_reg = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
plot_classification(
    tree_clf, data_clf[data_clf_columns], data_clf[target_clf_column],
    ax=axs[0]
)
plot_regression(
    tree_reg, data_reg[data_reg_columns], data_reg[target_reg_column],
    ax=axs[1])
axs[0].set_title(
    f"Optimal depth found via CV: {tree_clf.best_params_['max_depth']}"
)
axs[1].set_title(
    f"Optimal depth found via CV: {tree_reg.best_params_['max_depth']}"
)
plt.subplots_adjust(wspace=0.3)

# %% [markdown]
# The other parameters are used to fine tune the decision tree and have less
# impact than `max_depth`.
#
# # Main take away
#
# In this chapter, we presented decision tree in details. We saw that decision
#  trees:
#
# * are used in regression and classification problems;
# * are non-parametric models;
# * are not able to extrapolate;
# * are sensible to hyperparameter tuning.
