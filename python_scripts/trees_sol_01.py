# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M5.01
#
# In the previous notebook, we showed how a tree with 1 level depth works. The
# aim of this exercise is to repeat part of the previous experiment for a tree
# with 2 levels depth to show how such parameter affects the feature space
# partitioning.
#
# We first load the penguins dataset and split it into a training and a testing
# sets:

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
# training data.

# %%
# solution
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(data_train, target_train)

# %% [markdown]
# Now plot the data and the decision boundary of the trained classifier to see
# the effect of increasing the depth of the tree.
#
# Hint: Use the class `DecisionBoundaryDisplay` from the module
# `sklearn.inspection` as shown in previous course notebooks.
#
# ```{warning}
# At this time, it is not possible to use `response_method="predict_proba"` for
# multiclass problems. This is a planned feature for a future version of
# scikit-learn. In the mean time, you can use `response_method="predict"`
# instead.
# ```

# %%
# solution
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay


tab10_norm = mpl.colors.Normalize(vmin=-0.5, vmax=8.5)

palette = ["tab:blue", "tab:green", "tab:orange"]
DecisionBoundaryDisplay.from_estimator(
    tree,
    data_train,
    response_method="predict",
    cmap="tab10",
    norm=tab10_norm,
    alpha=0.5,
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
# except that for a K-class problem you have K probability outputs for each
# data point. Visualizing all these on a single plot can quickly become tricky
# to interpret. It is then common to instead produce K separate plots, one for
# each class, in a one-vs-rest (or one-vs-all) fashion.
#
# For example, in the plot below, the first plot on the left shows in yellow the
# certainty on classifying a data point as belonging to the "Adelie" class. In
# the same plot, the spectre from green to purple represents the certainty of
# **not** belonging to the "Adelie" class. The same logic applies to the other
# plots in the figure.

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

for class_of_interest in range(n_classes):
    axs[class_of_interest].set_title(
        f"Class {tree.classes_[class_of_interest]}"
    )
    imshow_handle = axs[class_of_interest].imshow(
        probas[:, class_of_interest].reshape((100, 100)),
        extent=(30, 60, 10, 23),
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        cmap="viridis",
    )
    axs[class_of_interest].set_xlabel("Culmen Length (mm)")
    if class_of_interest == 0:
        axs[class_of_interest].set_ylabel("Culmen Depth (mm)")
    idx = target_test == tree.classes_[class_of_interest]
    axs[class_of_interest].scatter(
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
# ```{note}
# You may have noticed that we are no longer using a diverging colormap. Indeed,
# the chance level for a one-vs-rest binarization of the multi-class
# classification problem is almost never at predicted probability of 0.5. So
# using a colormap with a neutral white at 0.5 might give a false impression on
# the certainty.
# ```
#
# In future versions of scikit-learn `DecisionBoundaryDisplay` will support a
# `class_of_interest` parameter that will allow in particular for a
# visualization of `predict_proba` in multi-class settings.
#
# We also plan to make it possible to visualize the `predict_proba` values for
# the class with the maximum predicted probability (without having to pass a
# given a fixed `class_of_interest` value).
