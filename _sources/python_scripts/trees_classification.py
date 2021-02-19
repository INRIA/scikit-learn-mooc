# %% [markdown]
# # Build a classification decision tree
#
# We will illustrate how decision tree fit data with a simple classification
# problem using the penguins dataset.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %% [markdown]
# Besides, we split the data into two subsets to investigate how trees will
# predict values based on an out-of-samples dataset.

# %%
from sklearn.model_selection import train_test_split

X, y = data[culmen_columns], data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
range_features = {
    feature_name: (X[feature_name].min() - 1, X[feature_name].max() + 1)
    for feature_name in X.columns}

# %% [markdown]
# In a previous notebook, we learnt that a linear classifier will define a
# linear separation to split classes using a linear combination of the input
# features. In our 2-dimensional space, it means that a linear classifier will
# define some oblique lines that best separate our classes. We define a
# function below that, given a set of data points and a classifier, will plot
# the decision boundaries learnt by the classifier.

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
# Thus, for a linear classifier, we will obtain the following decision
# boundaries. These boundaries lines indicate where the model changes its
# prediction from one class to another.

# %%
import seaborn as sns
from sklearn.linear_model import LogisticRegression
sns.set_context("talk")

# create a palette to be used in the scatterplot
palette = ["tab:red", "tab:blue", "black"]

linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

_, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=culmen_columns[0], y=culmen_columns[1], hue=target_column,
                data=data, palette=palette, ax=ax)
_ = plot_decision_function(linear_model, range_features, ax=ax)

# %% [markdown]
# We see that the lines are a combination of the input features since they are
# not perpendicular a specific axis. Indeed, this is due to the model
# parametrization that we saw in the previous notebook, controlled by the
# model's weights and intercept.
#
# Besides, it seems that the linear model would be a good candidate for
# such problem as it gives good accuracy.

# %%
model_name = linear_model.__class__.__name__
test_score = linear_model.fit(X_train, y_train).score(X_test, y_test)
print(f"Accuracy of the {model_name}: {test_score:.2f}")

# %% [markdown]
# Unlike linear models, decision trees are non-parametric models: they are not
# controlled by a mathematical decision function and do not have weights or
# intercept to be optimized.
#
# Indeed, decision trees will partition the space by considering a single
# feature at a time. Let's illustrate this behaviour by having a decision
# tree make a single split to partition the feature space.

# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X_train, y_train)

_, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=culmen_columns[0], y=culmen_columns[1], hue=target_column,
                data=data, palette=palette, ax=ax)
_ = plot_decision_function(tree, range_features, ax=ax)

# %% [markdown]
# The partitions found by the algorithm separates the data along the axis
# "Culmen Length", discarding the feature "Culmen Depth". Thus, it highlights
# that a decision tree does not use a combination of feature when making a
# split. We can look more in depth at the tree structure.

# %%
from sklearn.tree import plot_tree

_, ax = plt.subplots(figsize=(8, 6))
_ = plot_tree(tree, feature_names=culmen_columns,
              class_names=tree.classes_, impurity=False, ax=ax)

# %% [markdown]
# We see that the split was done the culmen length feature. The original
# dataset was subdivided into 2 sets based on the culmen length
# (inferior or superior to 16.45 mm).
#
# This partition of the dataset minimizes the class diversities in each
# sub-partitions. This measure is also known as a **criterion**,
# and is a settable parameter.
#
# If we look more closely at the partition, we see that the sample superior
# to 16.45 belongs mainly to the Adelie class. Looking at the values,
# we indeed observe 103 Adelie individuals in this space. 
# We also count 52 Chinstrap samples and 6 Gentoo samples.
# We can make similar interpretation for the partition defined by a threshold
# inferior to 16.45mm. In this case, the most represented class is the Gentoo
# species.
#
# Let's see how our tree would work as a predictor. Let's start to see the
# class predicted when the culmen length is inferior to the threshold.

# %%
tree.predict([[0, 15]])

# %% [markdown]
# The class predicted is the Gentoo. We can now check if we pass a culmen
# depth superior to the threshold.

# %%
tree.predict([[0, 17]])

# %% [markdown]
# In this case, the tree predicts the Adelie specie.
#
# Thus, we can conclude that a decision tree classifier will predict the most
# represented class within a partition.
#
# Since that during the training, we have a count of samples in each partition,
# we can also compute the probability of belonging to a specific class within
# this partition.

# %%
y_proba = pd.Series(tree.predict_proba([[0, 17]])[0],
                    index=tree.classes_)
ax = y_proba.plot(kind="bar")
ax.set_ylabel("Probability")
_ = ax.set_title("Probability to belong to a penguin class")

# %% [markdown]
# We will manually compute the different probability directly from the tree
# structure.

# %%
print(f"Probabilities for the different classes:\n"
      f"Adelie: {103 / 161:.3f}\n"
      f"Chinstrap: {52 / 161:.3f}\n"
      f"Gentoo: {6 / 161:.3f}\n")

# %% [markdown]
# It is also important to note that the culmen depth has been disregarded for
# the moment. It means that whatever the value given, it will not be used
# during the prediction.

# %%
tree.predict_proba([[10000, 17]])

# %% [markdown]
# Going back to our classification problem, the split found with a maximum
# depth of 1 is not powerful enough to separate the three species and the model
# accuracy is low when compared to the linear model.

# %%
model_name = tree.__class__.__name__
test_score = tree.fit(X_train, y_train).score(X_test, y_test)
print(f"Accuracy of the {model_name}: {test_score:.2f}")

# %% [markdown]
# Indeed, it is not a surprise. We saw earlier that a single feature will not
# be able to separate all three species. However, from the previous analysis we
# saw that by using both features we should be able to get fairly good results.
#
# In the next exercise, you will increase the size of the tree depth. You will
# get intuitions on how the space partitioning is repeated over time.
