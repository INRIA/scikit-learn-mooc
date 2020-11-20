# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Classification
# In regression, we saw that the target to be predicted was a continuous
# variable. In classification, this target will be discrete (e.g. categorical).
#
# We will go back to our penguin dataset. However, this time we will try to
# predict the penguin species using the culmen information. We will also
# simplify our classification problem by selecting only 2 of the penguin
# species to solve a binary classification problem.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins.csv")

# select the features of interest
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

data = data[culmen_columns + [target_column]]
data[target_column] = data[target_column].str.split().str[0]
data = data[data[target_column].apply(lambda x: x in ("Adelie", "Chinstrap"))]
data = data.dropna()

# %% [markdown]
# We can quickly start by visualizing the feature distribution by class:

# %%
import seaborn as sns
_ = sns.pairplot(data=data, hue="Species")

# %% [markdown]
# We can observe that we have quite a simple problem. When the culmen
# length increases, the probability that the penguin is a Chinstrap is closer
# to 1. However, the culmen depth is not helpful for predicting the penguin
# species.
#
# For model fitting, we will separate the target from the data and
# we will create a training and a testing set.

# %%
from sklearn.model_selection import train_test_split

X, y = data[culmen_columns], data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)

# %% [markdown]
# To visualize the separation found by our classifier, we will define an helper
# function `plot_decision_function`. In short, this function will fit our
# classifier and plot the edge of the decision function, where the probability
# to be an Adelie or Chinstrap will be equal (p=0.5).


# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(X, y, clf, title="auto", ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    clf.fit(X, y)

    # create a grid to evaluate all possible samples
    plot_step = 0.02
    feature_0_min, feature_0_max = (
        X.iloc[:, 0].min() - 1,
        X.iloc[:, 0].max() + 1,
    )
    feature_1_min, feature_1_max = (
        X.iloc[:, 1].min() - 1,
        X.iloc[:, 1].max() + 1,
    )
    xx, yy = np.meshgrid(
        np.arange(feature_0_min, feature_0_max, plot_step),
        np.arange(feature_1_min, feature_1_max, plot_step),
    )

    # compute the associated prediction
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(
        data=pd.concat([X, y], axis=1),
        x=X.columns[0],
        y=X.columns[1],
        hue=y.name,
        ax=ax,
    )
    if title == "auto":
        C = clf[-1].C if hasattr(clf[-1], "C") else clf[-1].C_
        ax.set_title(f"C={C}\n with coef={clf[-1].coef_[0]}")
    else:
        ax.set_title(title)


# %% [markdown]
# ### Un-penalized logistic regression
#
# The linear regression that we previously saw will predict a continuous
# output. When the target is a binary outcome, one can use the logistic
# function to model the probability. This model is known as logistic
# regression.
#
# Scikit-learn provides the class `LogisticRegression` which implements this
# algorithm.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="none")
)
plot_decision_function(X_train, y_train, logistic_regression)

# %% [markdown]
# Thus, we see that our decision function is represented by a line separating
# the 2 classes. Since the line is oblique, it means that we used a
# combination of both features:

# %%
print(logistic_regression[-1].coef_)

# %% [markdown]
# Indeed, both coefficients are non-null.
#
# ### Apply some regularization when fitting the logistic model
#
# The `LogisticRegression` model allows one to apply regularization via the
# parameter `C`. It would be equivalent to shifting from `LinearRegression`
# to `Ridge`. Ccontrary to `Ridge`, the value of the
# `C` parameter is inversely proportional to the regularization strength:
# a smaller `C` will lead to a more regularized model. We can check the effect
# of regularization on our model:

# %%
_, axs = plt.subplots(ncols=3, figsize=(12, 4))

for ax, C in zip(axs, [0.02, 0.1, 1]):
    logistic_regression = make_pipeline(
        StandardScaler(), LogisticRegression(C=C)
    )
    plot_decision_function(
        X_train, y_train, logistic_regression, ax=ax,
    )

# %% [markdown]
# A more regularized model will make the coefficients tend to 0. Since one of
# the features is considered less important when fitting the model (lower
# coefficient magnitude), only one of the feature will be used when C is small.
# This feature is the culmen length which is in line with our first insight
# when plotting the marginal feature probabilities.
#
# Just like the `RidgeCV` class which automatically finds the optimal `alpha`,
# one can use `LogisticRegressionCV` to find the best `C` on the training data.

# %%
from sklearn.linear_model import LogisticRegressionCV

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10])
)
plot_decision_function(X_train, y_train, logistic_regression)

# %% [markdown]
# ### Beyond linear separation
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
from sklearn.datasets import (
    make_moons, make_classification, make_gaussian_quantiles,
)

X_moons, y_moons = make_moons(n_samples=500, noise=.13, random_state=42)
X_class, y_class = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    random_state=2,
)
X_gauss, y_gauss = make_gaussian_quantiles(
    n_samples=50, n_features=2, n_classes=2, random_state=42,
)

datasets = [
    [pd.DataFrame(X_moons, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_moons, name="class")],
    [pd.DataFrame(X_class, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_class, name="class")],
    [pd.DataFrame(X_gauss, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_gauss, name="class")],
]

# %%
from sklearn.svm import SVC

_, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 9))

linear_model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
kernel_model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

for ax, (X, y) in zip(axs[0], datasets):
    plot_decision_function(X, y, linear_model, title="Linear kernel", ax=ax)
for ax, (X, y) in zip(axs[1], datasets):
    plot_decision_function(X, y, kernel_model, title="RBF kernel", ax=ax)

# %% [markdown]
# We see that the $R^2$ score decreases on each dataset, so we can say that each
# dataset is "less linearly separable" than the previous one.

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

# %%
