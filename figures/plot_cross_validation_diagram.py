from pathlib import Path
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


FIGURES_FOLDER = Path(__file__).parent
cmap_cv = plt.cm.coolwarm


def plot_cv_indices(cv, X, y, ax, lw=50):
    """Create a sample plot for indices of a cross-validation object."""
    splits = list(cv.split(X=X, y=y))
    n_splits = len(splits)

    # Generate the training/testing visualizations for each CV split
    for ii, (train, test) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.zeros(shape=X.shape[0], dtype=np.int32)
        indices[train] = 1

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(yticks=np.arange(n_splits) + .5,
           yticklabels=yticklabels, xlabel='Sample index',
           ylabel="CV iteration", ylim=[n_splits + .2,
                                        -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


n_points = 100
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, ax = plt.subplots(figsize=(20, 8))
cv = KFold(5)
_ = plot_cv_indices(cv, X, y, ax)
fig.savefig(FIGURES_FOLDER / "cross_validation_diagram.png")
plt.tight_layout()
