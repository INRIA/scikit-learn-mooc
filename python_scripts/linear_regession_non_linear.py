# %% [markdown]
# # Beyond linear separation in classification
#
# As we saw in regression, the linear classification model expects the data
# to be linearly separable. When this assumption does not hold, the model
# is not expressive enough to properly fit the data. One needs to apply the
# same tricks as in regression: feature augmentation (potentially using
# expert-knowledge) or using a kernel based method.
#
# We will provide examples where we will use a kernel support vector machine
# to perform classification on some toy-datasets where it is impossible to
# find a perfect linear separation.

# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(fitted_classifier, range_features, ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    feature_names = list(range_features.keys())
    # create a grid to evaluate all possible samples
    plot_step = 0.02
    xx, yy = np.meshgrid(
        np.arange(*range_features[feature_names[0]], plot_step),
        np.arange(*range_features[feature_names[1]], plot_step),
    )

    # compute the associated prediction
    Z = fitted_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

    return ax


# %%
import pandas as pd
from sklearn.datasets import (
    make_moons, make_classification, make_gaussian_quantiles,
)

X_moons, y_moons = make_moons(n_samples=500, noise=.13, random_state=42)
X_class, y_class = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    random_state=2,
)
X_gauss, y_gauss = make_gaussian_quantiles(
    n_samples=500, n_features=2, n_classes=2, random_state=42,
)

datasets = [
    [pd.DataFrame(X_moons, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_moons, name="class")],
    [pd.DataFrame(X_class, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_class, name="class")],
    [pd.DataFrame(X_gauss, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_gauss, name="class")],
]
range_features = {"feature #1": (-5, 5), "feature #2": (-5, 5)}

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 9))

linear_model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
kernel_model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

for ax, (X, y) in zip(axs[0], datasets):
    linear_model.fit(X, y)
    plot_decision_function(linear_model, range_features, ax=ax)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y,
                    palette=["tab:red", "tab:blue"], ax=ax)

for ax, (X, y) in zip(axs[1], datasets):
    kernel_model.fit(X, y)
    plot_decision_function(kernel_model, range_features, ax=ax)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y,
                    palette=["tab:red", "tab:blue"], ax=ax)

# %% [markdown]
# # Main take away
#
# - `LinearRegression` find the best slope which minimize the mean squared
#   error on the train set
# - `Ridge` could be better on the test set, thanks to its regularization
# - `RidgeCV` and `LogisiticRegressionCV` find the best relugarization thanks
#   to cross validation on the training data
# - `pipeline` can be used to combinate a scaler and a model
# - If the data are not linearly separable, we shall use a more complex model
#   or use feature augmentation
#
