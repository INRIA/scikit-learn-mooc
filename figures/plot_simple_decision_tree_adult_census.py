from pathlib import Path
import numpy as np

from scipy import ndimage

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


HERE = Path(__file__).parent

top = cm.get_cmap('Oranges', 128)
bottom = cm.get_cmap('Blues_r', 128)

colors = np.vstack([bottom(np.linspace(0, 1, 128)),
                    top(np.linspace(0, 1, 128))])
blue_orange_cmap = ListedColormap(colors, name='BlueOrange')


adult_census = pd.read_csv("../datasets/adult-census.csv")
target_column = 'class'

n_samples_to_plot = 5000

def plot_tree_decision_function(tree, X, y, ax=None):
    """Plot the different decision rules found by a `DecisionTreeClassifier`.

    Parameters
    ----------
    tree : DecisionTreeClassifier instance
        The decision tree to inspect.
    X : dataframe of shape (n_samples, n_features)
        The data used to train the `tree` estimator.
    y : ndarray of shape (n_samples,)
        The target used to train the `tree` estimator.
    ax : matplotlib axis
        The matplotlib axis where to plot the different decision rules.
    """
    import numpy as np
    from scipy import ndimage

    plt.figure(figsize=(12, 10))
    h = 0.02
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    faces = tree.tree_.apply(
        np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    faces = faces.reshape(xx.shape)
    border = ndimage.laplace(faces) != 0
    if ax is None:
        ax = plt.gca()
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1],
               c=np.array(['tab:blue', 'tab:orange'])[y],
               s=60, alpha=0.7,
               vmin=0, vmax=1)
    levels = np.linspace(0, 1, 101)
    contours = ax.contourf(xx, yy, Z, alpha=.4, levels=levels,
                           cmap=blue_orange_cmap)
    ax.get_figure().colorbar(contours, ticks=np.linspace(0, 1, 11))
    ax.scatter(xx[border], yy[border], marker='.', s=1)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    sns.despine(offset=10)
    plt.savefig(HERE / "simple_decision_tree_adult_census.png")


# select a subset of data
data_subset = adult_census[:n_samples_to_plot]
X = data_subset[["age", "hours-per-week"]]
y = LabelEncoder().fit_transform(
    data_subset[target_column].to_numpy())

max_leaf_nodes = 3
tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                              random_state=0)
tree.fit(X, y)

# plot the decision function learned by the tree
plot_tree_decision_function(tree, X, y)
