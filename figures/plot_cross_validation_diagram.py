import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.model_selection import KFold, ShuffleSplit


FIGURES_FOLDER = Path(__file__).parent
cmap_cv = plt.cm.coolwarm

plt.style.use(FIGURES_FOLDER / "../python_scripts/matplotlibrc")


# +
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
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=25,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, 50],
    )
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Training samples", "Testing samples"],
        loc=(1.02, 0.8),
    )
    ax.set_title("{}".format(type(cv).__name__))
    return ax


n_points = 50
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, ax = plt.subplots(figsize=(12, 4))
cv = KFold(5)
_ = plot_cv_indices(cv, X, y, ax)
plt.tight_layout()
fig.savefig(FIGURES_FOLDER / "cross_validation_diagram.png")

fig, ax = plt.subplots(figsize=(12, 4))
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
_ = plot_cv_indices(cv, X, y, ax)
plt.tight_layout()
fig.savefig(FIGURES_FOLDER / "shufflesplit_diagram.png")
