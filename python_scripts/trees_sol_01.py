# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M5.01
#
# In the previous notebook, we showed how a tree with a depth of 1 level was
# working. The aim of this exercise is to repeat part of the previous experiment
# for a depth with 2 levels to show how the process of partitioning is repeated
# over time.
#
# Before to start, we will:
#
# * load the dataset;
# * split the dataset into training and testing dataset;
# * define the function to show the classification decision function.

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
from sklearn.model_selection import train_test_split

data, target = penguins[culmen_columns], penguins[target_column]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0
)

# %% [markdown]
# Create a decision tree classifier with a maximum depth of 2 levels and fit the
# training data. Once this classifier trained, plot the data and the decision
# boundary to see the benefit of increasing the depth. To plot the decision
# boundary, you should import the class `DecisionBoundaryDisplay` from the
# module `sklearn.inspection` as shown in the previous course notebook.

# %%
# solution
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(data_train, target_train)

# %% tags=["solution"]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay

palette = ["tab:red", "tab:blue", "black"]
DecisionBoundaryDisplay.from_estimator(
    tree, data_train, response_method="predict", cmap="RdBu", alpha=0.5
)
ax = sns.scatterplot(
    data=penguins,
    x=culmen_columns[0],
    y=culmen_columns[1],
    hue=target_column,
    palette=palette,
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
_ = plt.title("Decision boundary using a decision tree")

# %% [markdown]
# Did we make use of the feature "Culmen Length"? Plot the tree using the
# function `sklearn.tree.plot_tree` to find out!

# %%
# solution
from sklearn.tree import plot_tree

_, ax = plt.subplots(figsize=(16, 12))
_ = plot_tree(
    tree,
    feature_names=culmen_columns,
    class_names=tree.classes_.tolist(),
    impurity=False,
    ax=ax,
)

# %% [markdown] tags=["solution"]
# The resulting tree has 7 nodes: 3 of them are "split nodes" and 4 are "leaf
# nodes" (or simply "leaves"), organized in 2 levels. We see that the second
# tree level used the "Culmen Length" to make two new decisions. Qualitatively,
# we saw that such a simple tree was enough to classify the penguins' species.

# %% [markdown]
# Compute the accuracy of the decision tree on the testing data.

# %%
# solution
test_score = tree.fit(data_train, target_train).score(data_test, target_test)
print(f"Accuracy of the DecisionTreeClassifier: {test_score:.2f}")

# %% [markdown] tags=["solution"]
# At this stage, we have the intuition that a decision tree is built by
# successively partitioning the feature space, considering one feature at a
# time.
#
# We predict an Adelie penguin if the feature value is below the threshold,
# which is not surprising since this partition was almost pure. If the feature
# value is above the threshold, we predict the Gentoo penguin, the class that is
# most probable.
#
# ## (Estimated) predicted probabilities in multi-class problems
#
# For those interested, one can further try to visualize the output of
# `predict_proba` for a multiclass problem using `DecisionBoundaryDisplay`,
# except that For a K-class problem, you will have K probability outputs for
# each data point. Visualizing all these on a single plot can quickly become
# tricky to interpret. It is then common to instead produce K separate plots,
# one for each class, in a one-vs-rest (or one-vs-all) fashion.
#
# For example, in the plot below, the first plot on the left shows in red the
# certainty on classifying a data point as belonging to the "Adelie" class. In
# the same plot, the blue color represents the certainty of **not** belonging to
# the "Adelie" class. The same logic applies to the other plots in the figure.

# %% tags=["solution"]
import numpy as np

xx = np.linspace(30, 60, 100)
yy = np.linspace(10, 23, 100)
xx, yy = np.meshgrid(xx, yy)
Xfull = pd.DataFrame(
    {"Culmen Length (mm)": xx.ravel(), "Culmen Depth (mm)": yy.ravel()}
)

probas = tree.predict_proba(Xfull)
n_classes = len(np.unique(tree.classes_))

_, axs = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(12, 5))
plt.suptitle("Predicted probabilities for decision tree model", y=0.8)

for k in range(n_classes):
    axs[k].set_title(f"Class {tree.classes_[k]}")
    imshow_handle = axs[k].imshow(
        probas[:, k].reshape((100, 100)),
        extent=(30, 60, 10, 23),
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        cmap="RdBu_r",
    )
    axs[k].set_xlabel("Culmen Length (mm)")
    if k == 0:
        axs[k].set_ylabel("Culmen Depth (mm)")
    idx = target_test == tree.classes_[k]
    axs[k].scatter(
        data_test["Culmen Length (mm)"].loc[idx],
        data_test["Culmen Depth (mm)"].loc[idx],
        marker="o",
        c="w",
        edgecolor="k",
    )

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")
_ = plt.title("Probability")

# %% [markdown] tags=["solution"]
# In scikit-learn v1.4 `DecisionBoundaryDisplay` will support a `class_of_interest`
# parameter that will allow in particular for a visualization of `predict_proba` in
# multi-class settings.
