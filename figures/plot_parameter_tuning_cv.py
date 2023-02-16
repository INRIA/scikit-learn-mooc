import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.model_selection import KFold


FIGURES_FOLDER = Path(__file__).parent
plt.style.use(FIGURES_FOLDER / "../python_scripts/matplotlibrc")

colors = ["#009e73ff", "#fd3c06ff", "#0072b2ff"]
cmap_name = "my_list"
cmap_cv = LinearSegmentedColormap.from_list(cmap_name, colors=colors, N=8)


def plot_cv_indices(cv, X, y, ax, lw=50):
    """Create a sample plot for indices of a cross-validation object
    embeded in a train-test split."""
    splits = list(cv.split(X=X, y=y))
    n_splits = len(splits)

    # Generate the training/testing visualizations for each CV split
    for ii, (train, test) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.zeros(shape=X.shape[0] + 10, dtype=np.int32)
        indices[train] = 1
        indices[X.shape[0] : X.shape[0] + 10] = 2

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=25,
            cmap=cmap_cv,
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
    ax.set_title(
        "{} cross validation inside (non-shuffled-)train-test split".format(
            type(cv).__name__
        )
    )
    ax.legend(
        [
            Patch(color=cmap_cv(0.9)),
            Patch(color=cmap_cv(0.5)),
            Patch(color=cmap_cv(0.02)),
        ],
        ["Testing samples", "Training samples", "Validation samples"],
        loc=(1.02, 0.7),
    )
    return ax


n_points = 40
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, ax = plt.subplots(figsize=(12, 4))
cv = KFold(5)
_ = plot_cv_indices(cv, X, y, ax)
plt.tight_layout()
fig.savefig(FIGURES_FOLDER / "cross_validation_train_test_diagram.png")


def plot_cv_nested_indices(cv_inner, cv_outer, X, y, ax, lw=50):
    """Create a sample plot for indices of a nested cross-validation object."""
    splits_outer = list(cv_outer.split(X=X, y=y))
    n_splits_outer = len(splits_outer)

    # Generate the training/testing visualizations for each CV split
    for ii, (train_outer, test_outer) in enumerate(splits_outer):

        splits_inner = list(cv_inner.split(train_outer))
        n_splits_inner = len(splits_inner)

        # Fill in indices with the training/test groups
        for jj, (train_inner, test_inner) in enumerate(splits_inner):
            indices = np.zeros(shape=X.shape[0], dtype=np.int32)
            indices[train_outer[train_inner]] = 1
            indices[test_outer] = 2

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [n_splits_inner * ii + jj + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=25,
                cmap=cmap_cv,
            )

    # Formatting
    ax.set_title("{} nested cross-validation".format(type(cv_outer).__name__))
    ax1 = ax.twinx()
    yticklabels = n_splits_outer * list(range(n_splits_inner))
    ax1.set(
        yticks=np.arange(n_splits_outer * n_splits_inner) + 0.3,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV inner iteration",
        ylim=[n_splits_outer * n_splits_inner + 0.2, -0.2],
        xlim=[0, 50],
    )
    yticklabels = list(range(n_splits_outer))
    ax.set(
        yticks=n_splits_inner*np.arange(n_splits_outer) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV outer iteration",
        ylim=[n_splits_outer * n_splits_inner + 0.2, 0.08],
        xlim=[0, 50],
    )
    ax.legend(
        [
            Patch(color=cmap_cv(0.9)),
            Patch(color=cmap_cv(0.5)),
            Patch(color=cmap_cv(0.02)),
        ],
        ["Testing samples", "Training samples", "Validation samples"],
        loc=(1.06, .93),
    )
    return ax


n_points = 50
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, ax = plt.subplots(figsize=(12, 12))
cv_inner = KFold(n_splits=4, shuffle=False)
cv_outer = KFold(n_splits=5, shuffle=False)
_ = plot_cv_nested_indices(cv_inner, cv_outer, X, y, ax)
plt.tight_layout()
fig.savefig(FIGURES_FOLDER / "nested_cross_validation_diagram.png")
