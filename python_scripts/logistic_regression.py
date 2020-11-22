# %% [markdown]
# # Linear model for classification
# In regression, we saw that the target to be predicted was a continuous
# variable. In classification, this target will be discrete (e.g. categorical).
#
# We will go back to our penguin dataset. However, this time we will try to
# predict the penguin species using the culmen information. We will also
# simplify our classification problem by selecting only 2 of the penguin
# species to solve a binary classification problem.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_classification.csv")

# only keep the Adelie and Chinstrap classes
data = data.set_index("Species").loc[["Adelie", "Chinstrap"]].reset_index()
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %% [markdown]
# We can quickly start by visualizing the feature distribution by class:

# %%
import seaborn as sns
sns.set_context("talk")

_ = sns.pairplot(data=data, hue="Species", height=3.3)

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
range_features = {
    feature_name: (X[feature_name].min() - 1, X[feature_name].max() + 1)
    for feature_name in X
}

# %% [markdown]
# To visualize the separation found by our classifier, we will define an helper
# function `plot_decision_function`. In short, this function will plot the edge
# of the decision function, where the probability to be an Adelie or Chinstrap
# will be equal (p=0.5).

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

    return ax


# %% [markdown]
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
logistic_regression.fit(X_train, y_train)

ax = plot_decision_function(logistic_regression, range_features)
_ = sns.scatterplot(
    x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_test,
    palette=["tab:red", "tab:blue"], ax=ax)

# %% [markdown]
# Thus, we see that our decision function is represented by a line separating
# the 2 classes. We should also note that we did not impose any regularization
# by setting the parameter `penalty` to `'none'`.
#
# Since the line is oblique, it means that we used a combination of both
# features:

# %%
weights = pd.Series(logistic_regression[-1].coef_.ravel(), index=X.columns)
_ = weights.plot(kind="barh")

# %% [markdown]
# Indeed, both coefficients are non-null.
