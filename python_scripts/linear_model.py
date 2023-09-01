# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
#
# # Non-linear feature engineering for Logistic Regression
#
#

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

feature_names = ["Feature #0", "Features #1"]
target_name = "class"

X, y = make_moons(n_samples=100, noise=0.13, random_state=42)

# We store both the data and target in a dataframe to ease plotting
moons = pd.DataFrame(
    np.concatenate([X, y[:, np.newaxis]], axis=1),
    columns=feature_names + [target_name],
)
data_moons, target_moons = moons[feature_names], moons[target_name]

# %%
from sklearn.datasets import make_gaussian_quantiles

feature_names = ["Feature #0", "Features #1"]
target_name = "class"

X, y = make_gaussian_quantiles(
    n_samples=100, n_features=2, n_classes=2, random_state=42
)
gauss = pd.DataFrame(
    np.concatenate([X, y[:, np.newaxis]], axis=1),
    columns=feature_names + [target_name],
)
data_gauss, target_gauss = gauss[feature_names], gauss[target_name]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

_, axs = plt.subplots(ncols=2, figsize=(14, 5))

sns.scatterplot(
    data=moons,
    x=feature_names[0],
    y=feature_names[1],
    hue=target_moons,
    palette=["tab:red", "tab:blue"],
    ax=axs[0],
)
sns.scatterplot(
    data=gauss,
    x=feature_names[0],
    y=feature_names[1],
    hue=target_gauss,
    palette=["tab:red", "tab:blue"],
    ax=axs[1],
)
axs[0].set_title("Illustration of the moons dataset")
_ = axs[1].set_title("Illustration of the Gaussian quantiles dataset")

# %%
from sklearn.inspection import DecisionBoundaryDisplay


def plot_decision_boundary(model):
    _, axs = plt.subplots(ncols=2, figsize=(14, 5))

    for ax, (data, target) in zip(
        axs,
        [
            (data_moons, target_moons),
            (data_gauss, target_gauss),
        ],
    ):
        model.fit(data, target)
        DecisionBoundaryDisplay.from_estimator(
            model,
            data,
            response_method="predict_proba",
            plot_method="pcolormesh",
            cmap="RdBu",
            alpha=0.8,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        DecisionBoundaryDisplay.from_estimator(
            model,
            data,
            response_method="predict_proba",
            plot_method="contour",
            alpha=0.8,
            levels=[0.5],
            linestyles="--",
            linewidths=2,
            ax=ax,
        )
        sns.scatterplot(
            data=data,
            x=feature_names[0],
            y=feature_names[1],
            hue=target,
            palette=["tab:red", "tab:blue"],
            ax=ax,
        )
    return axs


# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
logistic_regression

# %%
axs = plot_decision_boundary(logistic_regression)

# %%
from sklearn.preprocessing import KBinsDiscretizer

classifier = make_pipeline(KBinsDiscretizer(n_bins=8), LogisticRegression())
classifier

# %%
axs = plot_decision_boundary(classifier)

# %%
from sklearn.preprocessing import SplineTransformer

classifier = make_pipeline(
    SplineTransformer(n_knots=8),
    LogisticRegression(),
)
classifier

# %%
axs = plot_decision_boundary(classifier)

# %%
from sklearn.kernel_approximation import Nystroem

classifier = make_pipeline(
    Nystroem(kernel="rbf", gamma=1.0, n_components=100),
    LogisticRegression(C=10),
)
classifier

# %%
axs = plot_decision_boundary(classifier)

# %%
from sklearn.kernel_approximation import Nystroem

classifier = make_pipeline(
    SplineTransformer(n_knots=8),
    Nystroem(kernel="rbf", gamma=0.1, n_components=100),
    LogisticRegression(C=10),
)
classifier

# %%
axs = plot_decision_boundary(classifier)
# %%
