import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

from scipy import ndimage

from .tree_plotting import plot_tree as plot_tree_mpl

X, y = make_blobs(centers=[[0, 0], [1, 1]], random_state=61526, n_samples=50)


def plot_tree(max_depth=1):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    h = 0.02

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if max_depth != 0:
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
        tree.fit(X, y)
        Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        faces = tree.tree_.apply(
            np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        faces = faces.reshape(xx.shape)
        border = ndimage.laplace(faces) != 0
        ax[0].contourf(xx, yy, Z, alpha=.4, cmap='RdBu_r')
        ax[0].scatter(xx[border], yy[border], marker='.', s=1)
        ax[0].set_title("max_depth = %d" % max_depth)
        plot_tree_mpl(tree, ax=ax[1], impurity=False, filled=True)
        # ax[1].axis("off")
    else:
        ax[0].set_title("data set")
        ax[1].set_visible(False)
    ax[0].scatter(X[:, 0], X[:, 1], c=np.array(['b', 'r'])[y], s=60)
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    ax[0].set_xticks(())
    ax[0].set_yticks(())


def plot_tree_interactive():
    from ipywidgets import interactive, IntSlider
    slider = IntSlider(min=0, max=8, step=1, value=0)
    return interactive(plot_tree, max_depth=slider)
