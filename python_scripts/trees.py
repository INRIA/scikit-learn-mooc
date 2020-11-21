# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Decision tree in depth
#
# In this notebook, we will discuss in detail the internal algorithm used to
# build the decision tree. First, we will focus on the classification decision
# tree. Then, we will highlight the fundamental difference between the
# decision tree used for classification and regression. Finally, we will
# quickly discuss the importance of the hyper-parameters to be aware of when
# using decision trees.
#
# ## Presentation of the dataset
#
# We will use the
# [Palmer penguins dataset](https://allisonhorst.github.io/palmerpenguins/).
# This dataset is comprised of penguin records and ultimately, we want to
# predict the species each penguin belongs to.
#
# Each penguin is from one of the three following species: Adelie, Gentoo, and
# Chinstrap. See the illustration below depicting the three different penguin
# species:
#
# ![Image of penguins](https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/lter_penguins.png)
#
# This problem is a classification problem since the target is categorical.
# We will limit our input data to a subset of the original features
# to simplify our explanations when presenting the decision tree algorithm.
# Indeed, we will use feature based on penguins' culmen measurement. You can
# learn more about the penguins' culmen with illustration below:
#
# ![Image of culmen](https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/culmen_depth.png)

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_classification.csv")

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %% [markdown]
# Let's check the dataset more into details.

# %%
data.info()

# %% [markdown]
# We will separate the target from the data and create a training and a
# testing set.

# %%
from sklearn.model_selection import train_test_split

X, y = data[culmen_columns], data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)

# %% [markdown]
# Before going into detail about the decision tree algorithm, we will quickly
# inspect our dataset.

# %%
import seaborn as sns

_ = sns.pairplot(data=data, hue="Species")

# %% [markdown]
# We can first check the feature distributions by looking at the diagonal plots
# of the pairplot. We can build the following intuitions:
#
# * The Adelie species is separable from the Gentoo and Chinstrap species using
#   the culmen length;
# * The Gentoo species is separable from the Adelie and Chinstrap species using
#   the culmen depth.
#
# ## How are decision tree built?
#
# In a previous notebook, we learnt that a linear classifier will define a
# linear separation to split classes using a linear combination of the input
# features. In our 2-dimensional space, it means that a linear classifier will
# define some oblique lines that best separate our classes. We define a
# function below that, given a set of data points and a classifier, will plot
# the decision boundaries learnt by the classifier.

# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(X, y, clf, ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    clf.fit(X, y)

    # create a grid to evaluate all possible samples
    plot_step = 0.02
    feature_0_min, feature_0_max = (X.iloc[:, 0].min() - 1,
                                    X.iloc[:, 0].max() + 1)
    feature_1_min, feature_1_max = (X.iloc[:, 1].min() - 1,
                                    X.iloc[:, 1].max() + 1)
    xx, yy = np.meshgrid(
        np.arange(feature_0_min, feature_0_max, plot_step),
        np.arange(feature_1_min, feature_1_max, plot_step)
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
        x=X.columns[0], y=X.columns[1], hue=y.name,
        ax=ax,
    )


# %% [markdown]
# Thus, for a linear classifier, we will obtain the following decision
# boundaries. These boundaries lines indicate where the model changes its
# prediction from one class to another.

# %%
from sklearn.linear_model import LogisticRegression

linear_model = LogisticRegression()
plot_decision_function(X_train, y_train, linear_model)

# %% [markdown]
# We see that the lines are a combination of the input features since they are
# not perpendicular a specific axis. In addition, it seems that the linear
# model would be a good candidate model for such problem as it gives good
# accuracy.

# %%
print(
    f"Accuracy of the {linear_model.__class__.__name__}: "
    f"{linear_model.fit(X_train, y_train).score(X_test, y_test):.2f}"
)

# %% [markdown]
# Unlike linear models, decision trees will partition the space by considering
# a single feature at a time. Let's illustrate this behaviour by having
# a decision tree that only makes a single split to partition the feature
# space.

# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1)
plot_decision_function(X_train, y_train, tree)

# %% [markdown]
# The partitions found by the algorithm separates the data along the axis
# "Culmen Length", discarding the feature "Culmen Depth". Thus, it highlights
# that a decision tree does not use a combination of feature when making a
# split. We can look more in depth the tree structure.

# %%
from sklearn.tree import plot_tree

_, ax = plt.subplots(figsize=(8, 6))
_ = plot_tree(tree, ax=ax)

# %% [markdown]
# We see that the split was done the first feature `X[0]` (i.e. "Culmen
# Length"). The original dataset was subdivided into 2 sets depending if the
# culmen length was inferior or superior to 43.25 mm.
#
# This partition of the dataset is the one that minimize the class diversities
# in each sub-partitions. This measure is also known as called **criterion**
# and different criterion can be used when instantiating the decision tree.
# Here, it corresponds to the Gini impurity.
#
# If we look closely at the partition, the sample inferior to 43.25 belong
# mainly to the Adelie class. Looking at the tree structure, we indeed observe
# 109 Adelie samples. We also count 3 Chinstrap samples and 6 Gentoo samples.
# We can make similar interpretation for the partition defined by a threshold
# superior to 43.25 mm. In this case, the most represented class is the Gentoo
# specie.
#
# Let's see how our tree would work as a predictor. Let's start to see the
# class predicted when the culmen length is inferior to the threshold.

# %%
tree.predict([[40, 0]])

# %% [markdown]
# The class predicted is the Adelie. We can now check if we pass a culmen
# length superior to the threshold.

# %%
tree.predict([[50, 0]])

# %% [markdown]
# In this case, the tree predict the Gentoo specie.
#
# Thus, we can conclude that a decision tree classifier will predict the most
# represented class within a partition.
#
# Since that during the training, we have a count of samples in each partition,
# we can also compute a probability to belong to a certain class within this
# partition.

# %%
tree.predict_proba([[50, 0]])

# %% [markdown]
# We can manually compute the different probability directly from the tree
# structure

# %%
print(
    f"Probabilities for the different classes:\n"
    f"Adelie: {4 / 138:.3f}\n"
    f"Chinstrap: {48 / 138:.3f}\n"
    f"Gentoo: {86 / 138:.3f}\n"
)

# %% [markdown]
# It is also important to note that the culmen depth has been disregarded for
# the moment. It means that whatever the value given, it will not be used
# during the prediction.

# %%
tree.predict_proba([[50, 10000]])

# %% [markdown]
# Going back to our classification problem, the split found with a maximum
# depth of 1 is not powerful enough to separate the three species and the model
# accuracy is low when compared to the linear model.

# %%
print(
    f"Accuracy of the {tree.__class__.__name__}: "
    f"{tree.fit(X_train, y_train).score(X_test, y_test):.2f}"
)

# %% [markdown]
# Indeed, it is not a surprise. We saw earlier that a single feature will not
# be able to separate all three species. However, from the previous analysis we
# saw that by using both features we should be able to get fairly good results.
# Considering the splitting mechanism of the decision tree illustrated above,
# we should repeat the partitioning on the resulting rectangles created by the
# first split. In this regard, we expect that the two partitions at the second
# level of the tree will be using the feature "Culmen Depth".

# %%
tree = DecisionTreeClassifier(max_depth=2)
plot_decision_function(X_train, y_train, tree)

# %% [markdown]
# As expected, the decision tree made two new partitions using the "Culmen
# Depth". Now, our tree is more powerful with similar performance to our linear
# model.

# %%
print(
    f"Accuracy of the {tree.__class__.__name__}: "
    f"{tree.fit(X_train, y_train).score(X_test, y_test):.2f}"
)

# %% [markdown]
# At this stage, we have the intuition that a decision tree is built by
# successively partitioning the feature space, considering one feature at a
# time.


# %% [markdown]
# We predict an Adelie penguin if the feature value is below the threshold,
# which is not surprising since this partition was almost pure. If the feature
# value is above the threshold, we predict the Gentoo penguin, the class that
# is most probable.
#
# ## What about decision tree for regression?
#
# We explained the construction of the decision tree for a classification
# problem. In classification, we show that we minimized the class diversity. In
# regression, this criterion cannot be applied since `y` is continuous. To give
# some intuitions regarding the problem solved in regression, let's observe the
# characteristics of decision trees used for regression.
#
# ### Decision tree: a non-parametric model
#
# We will use the same penguins dataset however, this time we will formulate a
# regression problem instead of a classification problem. We will try to infer
# the body mass of a penguin given its flipper length.

# %%
data = pd.read_csv("../datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

X, y = data[data_columns], data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0,
)

# %%
sns.scatterplot(data=data, x="Flipper Length (mm)", y="Body Mass (g)")

# %% [markdown]
# Here, we deal with a regression problem because our target is a continuous
# variable ranging from 2.7 kg to 6.3 kg. From the scatter plot above, we can
# observe that we have a linear relationship between the flipper length
# and the body mass. The longer the flipper of a penguin, the heavier the
# penguin.
#
# For this problem, we would expect the simple linear model to be able to
# model this relationship.

# %%
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

# %% [markdown]
# We will first create a function in charge of plotting the dataset and
# all possible predictions. This function is equivalent to the earlier
# function used to plot the decision boundaries for classification.


# %%
def plot_regression_model(X, y, model, extrapolate=False, ax=None):
    """Plot the dataset and the prediction of a learnt regression model."""
    # train our model
    model.fit(X, y)

    # make a scatter plot of the input data and target
    training_data = pd.concat([X, y], axis=1)
    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(
        data=training_data, x="Flipper Length (mm)", y="Body Mass (g)",
        ax=ax, color="black", alpha=0.5,
    )

    # only necessary if we want to see the extrapolation of our model
    offset = 20 if extrapolate else 0

    # generate a testing set spanning between min and max of the training set
    X_test = np.linspace(
        X.min() - offset, X.max() + offset, num=100
    ).reshape(-1, 1)

    # predict for this testing set and plot the response
    y_pred = model.predict(X_test)
    ax.plot(
        X_test, y_pred,
        label=f"{model.__class__.__name__} trained", linewidth=3,
    )
    plt.legend()
    # return the axes in case we want to add something to it
    return ax


# %%
_ = plot_regression_model(X_train, y_train, linear_model)


# %% [markdown]
# On the plot above, we see that a non-regularized `LinearRegression` is able
# to fit the data. A feature of this model is that all new predictions
# will be on the line.

# %%
X_test_subset = X_test[:10]
ax = plot_regression_model(X_train, y_train, linear_model)
y_pred = linear_model.predict(X_test_subset)
ax.plot(
    X_test_subset, y_pred, label="Test predictions",
    color="tab:green", marker="^", markersize=10, linestyle="",
)

plt.legend()
# %% [markdown]
# Contrary to linear models, decision trees are non-parametric models, so they
# do not make assumptions about the way data are distributed. This will affect
# the prediction scheme. Repeating the above experiment will highlight the
# differences.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=1)

# %%
_ = plot_regression_model(X_train, y_train, tree)

# %% [markdown]
# We see that the decision tree model does not have a priori distribution for
# the data and we do not end-up with a straight line to regress flipper length
# and body mass.
#
# Instead, we observe that the predictions of the tree are piecewise constant.
# Indeed, our feature space was split into two partitions. We can check the
# tree structure to see what was the threshold found during the training.

# %%
_ = plot_tree(tree)

# %% [markdown]
# The threshold for our feature (flipper length) is 206.5 mm. The predicted
# values on each side of the split are two constants: 3686.29 g and 5025.99 g.
# These values corresponds to the mean values of the training samples in each
# partition.
#
# Increasing the depth of the tree will increase the number of partition and
# thus the number of constant values that the tree is capable of predicting.

# %%
tree = DecisionTreeRegressor(max_depth=3)
_ = plot_regression_model(X_train, y_train, tree)

# %% [markdown]
# This lead us to question whether or not our decision trees are able to
# extrapolate to unseen data. We can highlight that this is possible with the
# linear model because it is a parametric model.

# %%
plot_regression_model(X_train, y_train, linear_model, extrapolate=True)

# %% [markdown]
# The linear model will extrapolate using the fitted model for flipper lengths
# < 175 mm and > 235 mm. Let's see the difference between the classification
# and regression trees.

# %%
ax = plot_regression_model(X_train, y_train, linear_model, extrapolate=True)
_ = plot_regression_model(X_train, y_train, tree, extrapolate=True, ax=ax)

# %% [markdown]
# For the regression tree, we see that it cannot extrapolate outside of the
# flipper length range present in the training data.
# For flipper lengths below the minimum, the mass of the penguin in the
# training data with the shortest flipper length will always be predicted.
# Similarly, for flipper lengths above the maximum, the mass of the penguin
# in the training data with the longest flipper will always predicted.
#
# ## Importance of decision tree hyper-parameters on generalization
#
# This last section will illustrate the importance of some key hyper-parameters
# of the decision tree. We will illustrate it on both the classification and
# regression probelms that we previously used.
#
# ### Creation of the classification and regression dataset
#
# We will first regenerate the classification and regression dataset.

# %%
data_clf = pd.read_csv("../datasets/penguins_classification.csv")

# %%
data_clf_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_clf_column = "Species"

X_clf, y_clf = data_clf[data_clf_columns], data_clf[target_clf_column]
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, stratify=y_clf, random_state=0,
)

# %%
data_reg_columns = ["Flipper Length (mm)"]
target_reg_column = "Body Mass (g)"

data_reg = pd.read_csv("../datasets/penguins_regression.csv")

X_reg, y_reg = data_reg[data_reg_columns], data_reg[target_reg_column]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, random_state=0,
)

# %%
_, axs = plt.subplots(ncols=2, figsize=(10, 5))
sns.scatterplot(
    data=data_clf,
    x="Culmen Length (mm)", y="Culmen Depth (mm)", hue="Species",
    ax=axs[0],
)
axs[0].set_title("Classification dataset")
sns.scatterplot(
    data=data_reg, x="Flipper Length (mm)", y="Body Mass (g)",
    ax=axs[1],
)
_ = axs[1].set_title("Regression dataset")

# %% [markdown]
# ### Effect of the `max_depth` parameter
#
# In decision trees, the most important parameter to get a trade-off between
# under-fitting and over-fitting is the `max_depth` parameter. Let's build
# a shallow tree and then deeper tree (for both classification and regression).


# %%
max_depth = 2
tree_clf = DecisionTreeClassifier(max_depth=max_depth)
tree_reg = DecisionTreeRegressor(max_depth=max_depth)

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
plot_decision_function(X_train_clf, y_train_clf, tree_clf, ax=axs[0])
plot_regression_model(X_train_reg, y_train_reg, tree_reg, ax=axs[1])
_ = fig.suptitle(f"Shallow tree with a max-depth of {max_depth}")


# %%
max_depth = 30
tree_clf = DecisionTreeClassifier(max_depth=max_depth)
tree_reg = DecisionTreeRegressor(max_depth=max_depth)

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
plot_decision_function(X_train_clf, y_train_clf, tree_clf, ax=axs[0])
plot_regression_model(X_train_reg, y_train_reg, tree_reg, ax=axs[1])
_ = fig.suptitle(f"Deep tree with a max-depth of {max_depth}")

# %% [markdown]
# For both classification and regression setting, we can observe that
# increasing the depth will make the tree model more expressive. However, a
# tree that is too deep will overfit the training data, creating partitions
# which are only be correct for "outliers". The `max_depth` is one of the
# hyper-parameters that one should optimize via cross-validation and
# grid-search.

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": np.arange(2, 10, 1)}
tree_clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
tree_reg = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
plot_decision_function(X_train_clf, y_train_clf, tree_clf, ax=axs[0])
axs[0].set_title(
    f"Optimal depth found via CV: {tree_clf.best_params_['max_depth']}"
)
plot_regression_model(X_train_reg, y_train_reg, tree_reg, ax=axs[1])
_ = axs[1].set_title(
    f"Optimal depth found via CV: {tree_reg.best_params_['max_depth']}"
)

# %% [markdown]
# The other parameters are used to fine tune the decision tree and have less
# impact than `max_depth`.
