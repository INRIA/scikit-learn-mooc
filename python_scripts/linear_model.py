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

    model.fit(data_moons, target_moons)
    DecisionBoundaryDisplay.from_estimator(
        model,
        data_moons,
        response_method="predict_proba",
        plot_method="pcolormesh",
        # response_method="predict",
        cmap="RdBu",
        alpha=0.5,
        ax=axs[0],
    )
    sns.scatterplot(
        data=moons,
        x=feature_names[0],
        y=feature_names[1],
        hue=target_moons,
        palette=["tab:red", "tab:blue"],
        ax=axs[0],
    )

    model.fit(data_gauss, target_gauss)
    DecisionBoundaryDisplay.from_estimator(
        model,
        data_gauss,
        response_method="predict_proba",
        plot_method="pcolormesh",
        # response_method="predict",
        cmap="RdBu",
        alpha=0.5,
        ax=axs[1],
    )
    sns.scatterplot(
        data=gauss,
        x=feature_names[0],
        y=feature_names[1],
        hue=target_gauss,
        palette=["tab:red", "tab:blue"],
        ax=axs[1],
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

logistic_regression = make_pipeline(
    StandardScaler(), KBinsDiscretizer(), LogisticRegression()
)
logistic_regression

# %%
axs = plot_decision_boundary(logistic_regression)

# %%
from sklearn.preprocessing import SplineTransformer

logistic_regression = make_pipeline(
    StandardScaler(),
    SplineTransformer(include_bias=False),
    LogisticRegression(),
)
logistic_regression

# %%
axs = plot_decision_boundary(logistic_regression)

# %%
from sklearn.preprocessing import PolynomialFeatures

logistic_regression = make_pipeline(
    KBinsDiscretizer(n_bins=3),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    LogisticRegression(),
)
logistic_regression

# %%
axs = plot_decision_boundary(logistic_regression)

# %%
from sklearn.kernel_approximation import Nystroem

logistic_regression = make_pipeline(
    StandardScaler(),
    SplineTransformer(extrapolation="linear", include_bias=False),
    Nystroem(kernel="rbf", gamma=5, n_components=100),
    LogisticRegression(C=5),
)
logistic_regression

# %%
axs = plot_decision_boundary(logistic_regression)

# %%
