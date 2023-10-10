# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Build a classification decision tree
#
# In this notebook we illustrate decision trees in a multiclass classification
# problem by using the penguins dataset with 2 features and 3 classes.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %% [markdown]
# First, we split the data into two subsets to investigate how trees predict
# values based on unseen data.

# %%
from sklearn.model_selection import train_test_split

data, target = penguins[culmen_columns], penguins[target_column]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0
)

# %% [markdown]
# In a previous notebook, we learnt that linear classifiers define a linear
# separation to split classes using a linear combination of the input features.
# In our 2-dimensional feature space, it means that a linear classifier finds
# the oblique lines that best separate the classes. This is still true for
# multiclass problems, except that more than one line is fitted. We can use
# `DecisionBoundaryDisplay` to plot the decision boundaries learnt by the
# classifier.

# %%
from sklearn.linear_model import LogisticRegression

linear_model = LogisticRegression()
linear_model.fit(data_train, target_train)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay

# create a palette to be used in the scatterplot
palette = ["tab:red", "tab:blue", "black"]

DecisionBoundaryDisplay.from_estimator(
    linear_model, data_train, response_method="predict", cmap="RdBu", alpha=0.5
)
sns.scatterplot(
    data=penguins,
    x=culmen_columns[0],
    y=culmen_columns[1],
    hue=target_column,
    palette=palette,
)
# put the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
_ = plt.title("Decision boundary using a logistic regression")

# %% [markdown]
# We see that the lines are a combination of the input features since they are
# not perpendicular a specific axis. Indeed, this is due to the model
# parametrization that we saw in some previous notebooks, i.e. controlled by the
# model's weights and intercept.
#
# Besides, it seems that the linear model would be a good candidate for such
# problem as it gives good accuracy.

# %%
linear_model.fit(data_train, target_train)
test_score = linear_model.score(data_test, target_test)
print(f"Accuracy of the LogisticRegression: {test_score:.2f}")

# %% [markdown]
# Unlike linear models, decision trees are non-parametric models: they are not
# controlled by a mathematical decision function and do not have weights or an
# intercept to be optimized.
#
# Indeed, decision trees partition the space by considering a single feature at
# a time. Let's illustrate this behaviour by having a decision tree make a
# single split to partition the feature space.

# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(data_train, target_train)

# %%
DecisionBoundaryDisplay.from_estimator(
    tree, data_train, response_method="predict", cmap="RdBu", alpha=0.5
)
sns.scatterplot(
    data=penguins,
    x=culmen_columns[0],
    y=culmen_columns[1],
    hue=target_column,
    palette=palette,
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
_ = plt.title("Decision boundary using a decision tree")

# %% [markdown]
# The partitions found by the algorithm separates the data along the axis
# "Culmen Depth", discarding the feature "Culmen Length". Thus, it highlights
# that a decision tree does not use a combination of features when making a
# split. We can look more in depth at the tree structure.

# %%
from sklearn.tree import plot_tree

_, ax = plt.subplots(figsize=(8, 6))
_ = plot_tree(
    tree,
    feature_names=culmen_columns,
    class_names=tree.classes_.tolist(),
    impurity=False,
    ax=ax,
)

# %% [markdown]
# ```{tip}
# We are using the function `fig, ax = plt.subplots(figsize=(8, 6))` to create
# a figure and an axis with a specific size. Then, we can pass the axis to the
# `sklearn.tree.plot_tree` function such that the drawing happens in this axis.
# ```

# %% [markdown]
# We see that the split was done on the culmen depth feature. The original
# dataset was subdivided into 2 sets based on the culmen depth (inferior or
# superior to 16.45 mm).
#
# This partition of the dataset minimizes the class diversity in each
# sub-partitions. This measure is also known as a **criterion**, and is a
# settable parameter.
#
# If we look more closely at the partition, we see that the sample superior to
# 16.45 belongs mainly to the "Adelie" class. Looking at the values, we indeed
# observe 103 "Adelie" individuals in this space. We also count 52 "Chinstrap"
# samples and 6 "Gentoo" samples. We can make similar interpretation for the
# partition defined by a threshold inferior to 16.45mm. In this case, the most
# represented class is the "Gentoo" species.
#
# Let's see how our tree would work as a predictor. Let's start with a case
# where the culmen depth is inferior to the threshold.

# %%
test_penguin_1 = pd.DataFrame(
    {"Culmen Length (mm)": [0], "Culmen Depth (mm)": [15]}
)
tree.predict(test_penguin_1)

# %% [markdown]
# The class predicted is the "Gentoo". We can now check what happens if we pass a
# culmen depth superior to the threshold.

# %%
test_penguin_2 = pd.DataFrame(
    {"Culmen Length (mm)": [0], "Culmen Depth (mm)": [17]}
)
tree.predict(test_penguin_2)

# %% [markdown]
# In this case, the tree predicts the "Adelie" specie.
#
# Thus, we can conclude that a decision tree classifier predicts the most
# represented class within a partition.
#
# During the training, we have a count of samples in each partition, we can also
# compute the probability of belonging to a specific class within this
# partition.

# %%
y_pred_proba = tree.predict_proba(test_penguin_2)
y_proba_class_0 = pd.Series(y_pred_proba[0], index=tree.classes_)

# %%
y_proba_class_0.plot.bar()
plt.ylabel("Probability")
_ = plt.title("Probability to belong to a penguin class")

# %% [markdown]
# We can also compute the different probabilities manually directly from the
# tree structure.

# %%
adelie_proba = 103 / 161
chinstrap_proba = 52 / 161
gentoo_proba = 6 / 161
print(
    "Probabilities for the different classes:\n"
    f"Adelie: {adelie_proba:.3f}\n"
    f"Chinstrap: {chinstrap_proba:.3f}\n"
    f"Gentoo: {gentoo_proba:.3f}\n"
)

# %% [markdown]
# It is also important to note that the culmen length has been disregarded for
# the moment. It means that regardless of its value, it is not be used during
# the prediction.

# %%
test_penguin_3 = pd.DataFrame(
    {"Culmen Length (mm)": [10_000], "Culmen Depth (mm)": [17]}
)
tree.predict_proba(test_penguin_3)

# %% [markdown]
# Going back to our classification problem, the split found with a maximum depth
# of 1 is not powerful enough to separate the three species and the model
# accuracy is low when compared to the linear model.

# %%
tree.fit(data_train, target_train)
test_score = tree.score(data_test, target_test)
print(f"Accuracy of the DecisionTreeClassifier: {test_score:.2f}")

# %% [markdown]
# Indeed, it is not a surprise. We saw earlier that a single feature is not able
# to separate all three species: it underfits. However, from the previous
# analysis we saw that by using both features we should be able to get fairly
# good results.
#
# In the next exercise, you will increase the tree depth to get an intuition on
# how such parameter affects the space partitioning.
#
# Finally, we can try to visualize the output of predict_proba for a multiclass
# problem using `DecisionBoundaryDisplay`, except that For a K-class problem,
# you'll have K probability outputs for each data point. Visualizing all these
# on a single plot can quickly become tricky to interpret. It is then common to
# instead produce K separate plots, one for each class, in a one-vs-rest (or
# one-vs-all) fashion.
#
# For example, in the plot below, the first column shows in red the certainty on
# classifying a data point as belonging to the "Adelie" class. Notice that the
# logistic regression is more certain than our under-fitting tree in this case.
# Indeed, the shallow tree is unsure between classes "Adelie" and "Chinstrap".
# In the same column, the blue color represents the certainty of **not**
# belonging to the "Adelie" class. The same logic applies to the other columns.

# %%
import numpy as np

classifiers = {
    "logistic": linear_model,
    "tree": tree,
}
n_classifiers = len(classifiers)

xx = np.linspace(30, 60, 100)
yy = np.linspace(10, 23, 100)
xx, yy = np.meshgrid(xx, yy)
Xfull = pd.DataFrame(
    {"Culmen Length (mm)": xx.ravel(), "Culmen Depth (mm)": yy.ravel()}
)

plt.figure(figsize=(12, 4))
plt.subplots_adjust(bottom=0.2, top=0.95)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(data_train, target_train)
    target_pred = classifier.predict(data_test)
    probas = classifier.predict_proba(Xfull)
    n_classes = len(np.unique(classifier.classes_))

    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
        plt.title(f"Class {classifier.classes_[k]}")
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(
            probas[:, k].reshape((100, 100)),
            extent=(30, 60, 10, 23),
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            cmap="RdBu_r",
        )
        plt.xticks(())
        plt.yticks(())
        idx = target_test == classifier.classes_[k]
        plt.scatter(
            data_test["Culmen Length (mm)"].loc[idx],
            data_test["Culmen Depth (mm)"].loc[idx],
            marker="o",
            c="w",
            edgecolor="k",
        )

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")
_ = plt.title("Probability")

# %% [markdown]
# In scikit-learn v1.4 `DecisionBoundaryDisplay` will support a `class_of_interest`
# parameter that will allow in particular for a visualization of `predict_proba` in
# multi-class settings.
