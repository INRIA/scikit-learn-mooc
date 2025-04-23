import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.model_selection import KFold


FIGURES_FOLDER = Path(__file__).parent
plt.style.use(FIGURES_FOLDER / "../python_scripts/matplotlibrc")

colors_cv = ["#009e73ff", "#fd3c06ff", "white"]
colors_eval = ["#fd3c06ff", "#fd3c06ff", "#0072b2ff"]
cmap_cv = ListedColormap(colors=colors_cv)
cmap_eval = ListedColormap(colors=colors_eval)


def plot_cv_indices(cv, X, y, axs):
    """Create a sample plot for indices of a cross-validation object
    embeded in a train-test split."""
    splits = list(cv.split(X=X, y=y))
    n_splits = len(splits)
    ax1, ax2 = axs

    # Generate the training/testing visualizations for each CV split
    for ii, (train, test) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.zeros(shape=X.shape[0] + 10, dtype=np.int32)
        indices[train] = 1
        indices[X.shape[0] : X.shape[0] + 10] = 2

        # Visualize the results
        ax1.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=25,
            cmap=cmap_cv,
        )
    ax2.scatter(
        range(len(indices)),
        [0.5] * len(indices),
        c=indices,
        marker="_",
        lw=25,
        cmap=cmap_eval,
    )

    # Formatting
    yticklabels = list(range(n_splits))
    ax1.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        ylabel="CV iteration",
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, 50],
    )

    ax2.set(
        yticks=[0.5],
        yticklabels=[],
        xlabel="Sample index",
        ymargin=10,
        ylim=[0.3, 0.7],
        xlim=[0, 50],
    )
    ax2.set_ylabel("refit +\nevaluation", labelpad=15)
    ax2.legend(
        [
            Patch(color=cmap_cv(0.5)),
            Patch(color=cmap_cv(0.02)),
            Patch(color=cmap_eval(0.9)),
        ],
        [
            "Training samples",
            "Validation samples\n(for hyperparameter\ntuning)",
            "Testing samples\n(reserved until\nfinal evaluation)",
        ],
        loc=(1.02, 1.1),
        labelspacing=1,
    )
    return


n_points = 40
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, axs = plt.subplots(
    ncols=1,
    nrows=2,
    sharex=True,
    figsize=(12, 5),
    gridspec_kw={"height_ratios": [5, 1.5], "hspace": 0},
)
cv = KFold(5)
plot_cv_indices(cv, X, y, axs)
plt.suptitle(
    "Internal {} cross-validation in GridSearchCV".format(type(cv).__name__),
    y=0.95,
)
plt.tight_layout()
fig.savefig(FIGURES_FOLDER / "cross_validation_train_test_diagram.png")


def plot_cv_nested_indices(cv_inner, cv_outer, X, y, axs):
    """Create a sample plot for indices of a nested cross-validation object."""
    splits_outer = list(cv_outer.split(X=X, y=y))
    n_splits_outer = len(splits_outer)

    # Generate the training/testing visualizations for each CV split
    for outer_index, (train_outer, test_outer) in enumerate(splits_outer):
        splits_inner = list(cv_inner.split(train_outer))
        n_splits_inner = len(splits_inner)

        # Fill in indices with the training/test groups
        for inner_index, (train_inner, test_inner) in enumerate(splits_inner):
            indices = np.zeros(shape=X.shape[0], dtype=np.int32)
            indices[train_outer[train_inner]] = 1
            indices[test_outer] = 2

            # Visualize the results
            axs[outer_index * 2].scatter(
                range(len(indices)),
                [inner_index + 0.6] * len(indices),
                c=indices,
                marker="_",
                lw=25,
                cmap=cmap_cv,
            )

        axs[outer_index * 2 + 1].scatter(
            range(len(indices)),
            [0.5] * len(indices),
            c=indices,
            marker="_",
            lw=25,
            cmap=cmap_eval,
        )
        axs[outer_index * 2 + 1].set(
            yticks=[0.5],
            yticklabels=["refit +\nevaluation"],
            xlabel="Sample index",
            ymargin=10,
            ylim=[0.3, 0.7],
            xlim=[0, 50],
        )

        # Formatting
        ax_twin = axs[outer_index * 2].twinx()
        yticklabels = list(range(n_splits_inner))
        ax_twin.set(
            yticks=np.arange(n_splits_inner) + 0.4,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylabel="inner iteration",
            ylim=[n_splits_inner + 0.2, -0.2],
            xlim=[0, 50],
        )

        axs[outer_index * 2].set(
            yticks=n_splits_inner * np.arange(n_splits_outer) + 0.5,
            yticklabels=[outer_index] * n_splits_outer,
            xlabel="Sample index",
            ylim=[n_splits_inner + 0.2, 0.08],
            xlim=[0, 50],
        )

    axs[0].legend(
        [
            Patch(color=cmap_cv(0.5)),
            Patch(color=cmap_cv(0.02)),
            Patch(color=cmap_eval(0.9)),
        ],
        [
            "Training samples",
            "Validation samples\n(for hyperparameter\ntuning)",
            "Testing samples\n(reserved until\nevaluation)",
        ],
        loc=(1.2, -0.2),
        labelspacing=1,
    )
    return


n_points = 50
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, axs = plt.subplots(
    ncols=1,
    nrows=10,
    sharex=True,
    figsize=(14, 15),
    gridspec_kw={"height_ratios": [5, 1.5] * 5, "hspace": 0},
)
cv_inner = KFold(n_splits=4, shuffle=False)
cv_outer = KFold(n_splits=5, shuffle=False)
plot_cv_nested_indices(cv_inner, cv_outer, X, y, axs)
plt.suptitle(
    "{} nested cross-validation".format(type(cv_outer).__name__), y=0.97
)
plt.tight_layout()
fig.text(0.0, 0.5, "outer iteration", va="center", rotation="vertical")
fig.savefig(FIGURES_FOLDER / "nested_cross_validation_diagram.png")
