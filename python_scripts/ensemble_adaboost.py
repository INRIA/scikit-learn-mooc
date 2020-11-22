# %% [markdown]
# # Adaptive Boosting (AdaBoost)
#
# In this notebook, we present the Adaptive Boosting (AdaBoost) algorithm. The
# aim is to intuitions regarding the internal machinery of AdaBoost and
# boosting more in general.
#
#  We will load the "penguin" dataset used in the "tree in depth" notebook. We
# will predict penguin species from the features culmen length and depth.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

X, y = data[culmen_columns], data[target_column]
range_features = {
    feature_name: (X[feature_name].min() - 1, X[feature_name].max() + 1)
    for feature_name in X.columns
}

# %% [markdown]
# In addition, we are also using on the function used the previous "tree in
# depth" notebook to plot the decision function of the tree.

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
# We will purposely train a shallow decision tree. Since the tree is shallow,
# it is unlikely to overfit and some of the training examples will even be
# misclassified on the training set.

# %%
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
sns.set_context("talk")

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X, y)

_, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=culmen_columns[0], y=culmen_columns[1], hue=target_column,
    data=data, palette=["tab:red", "tab:blue", "black"], ax=ax)
_ = plot_decision_function(tree, range_features, ax=ax)

# find the misclassified samples
y_pred = tree.predict(X)
misclassified_samples_idx = np.flatnonzero(y != y_pred)

ax.plot(X.iloc[misclassified_samples_idx, 0],
        X.iloc[misclassified_samples_idx, 1],
        "+k", label="Misclassified samples")
_ = ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

# %% [markdown]
# We observe that several samples have been misclassified by the classifier.
#
# We mentioned that boosting relies on creating a new classifier which tries to
# correct these misclassifications. In scikit-learn, learners support a
# parameter `sample_weight` which forces the learner to pay more attention to
# samples with higher weights, during the training.
#
# This parameter is set when calling
# `classifier.fit(X, y, sample_weight=weights)`.
# We will use this trick to create a new classifier by 'discarding' all
# correctly classified samples and only considering the misclassified samples.
# Thus, misclassified samples will be assigned a weight of 1 while well
# classified samples will assigned to a weight of 0.

# %%
sample_weight = np.zeros_like(y, dtype=int)
sample_weight[misclassified_samples_idx] = 1

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X, y, sample_weight=sample_weight)

_, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=culmen_columns[0], y=culmen_columns[1], hue=target_column,
    data=data, palette=["tab:red", "tab:blue", "black"], ax=ax)
plot_decision_function(tree, range_features, ax=ax)

ax.plot(X.iloc[misclassified_samples_idx, 0],
        X.iloc[misclassified_samples_idx, 1],
        "+k", label="Previous misclassified samples")
_ = ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

# %% [markdown]
# We see that the decision function drastically changed. Qualitatively, we see
# that the previously misclassified samples are now correctly classified.

# %%
y_pred = tree.predict(X)
newly_misclassified_samples_idx = np.flatnonzero(y != y_pred)
remaining_misclassified_samples_idx = np.intersect1d(
    misclassified_samples_idx, newly_misclassified_samples_idx
)

print(
    f"Number of samples previously misclassified and still misclassified: "
    f"{len(remaining_misclassified_samples_idx)}"
)

# %% [markdown]
# However, we are making mistakes on previously well classified samples. Thus,
# we get the intuition that we should weight the predictions of each classifier
# differently, most probably by using the number of mistakes each classifier
# is making.
#
# So we could use the classification error to combine both trees.

# %%
ensemble_weight = [
    (y.shape[0] - len(misclassified_samples_idx)) / y.shape[0],
    (y.shape[0] - len(newly_misclassified_samples_idx)) / y.shape[0],
]
ensemble_weight

# %% [markdown]
# The first classifier was 94% accurate and the second one 69% accurate.
# Therefore, when predicting a class, we should trust the first classifier
# slightly more than the second one. We could use these accuracy values to
# weight the predictions of each learner.
#
# To summarize, boosting learns several classifiers, each of which will
# focus more or less on specific samples of the dataset. Boosting is thus
# different from bagging: here we never resample our dataset, we just assign
# different weights to the original dataset.
#
# Boosting requires some strategy to combine the learners together:
#
# * one needs to define a way to compute the weights to be assigned
#   to samples;
# * one needs to assign a weight to each learner when making predictions.
#
# Indeed, we defined a really simple scheme to assign sample weights and
# learner weights. However, there are statistical theories (like in AdaBoost)
# for how these sample and learner weights can be optimally calculated.
#
# **FIXME: I think we should add a reference to ESL here.**
#
# We will use the AdaBoost classifier implemented in scikit-learn and
# look at the underlying decision tree classifiers trained.

# %%
from sklearn.ensemble import AdaBoostClassifier

base_estimator = DecisionTreeClassifier(max_depth=3, random_state=0)
adaboost = AdaBoostClassifier(
    base_estimator=base_estimator, n_estimators=3, algorithm="SAMME",
    random_state=0)
adaboost.fit(X, y)

_, axs = plt.subplots(ncols=3, figsize=(18, 6))

for ax, tree in zip(axs, adaboost.estimators_):
    sns.scatterplot(
        x=culmen_columns[0], y=culmen_columns[1], hue=target_column,
        data=data, palette=["tab:red", "tab:blue", "black"], ax=ax)
    plot_decision_function(tree, range_features, ax=ax)
plt.subplots_adjust(wspace=0.35)

print(f"Weight of each classifier: {adaboost.estimator_weights_}")
print(f"Error of each classifier: {adaboost.estimator_errors_}")

# %% [markdown]
# We see that AdaBoost has learnt three different classifiers each of which
# focuses on different samples. Looking at the weights of each learner, we see
# that the ensemble gives the highest weight to the first classifier. This
# indeed makes sense when we look at the errors of each classifier. The first
# classifier also has the highest classification performance.
#
# While AdaBoost is a nice algorithm to demonsrate the internal machinery of
# boosting algorithms, it is not the most efficient machine-learning algorithm.
# The most efficient algorithm based on boosting is the gradient-boosting
# decision tree (GBDT) algorithm which we will discuss after a short exercise.
