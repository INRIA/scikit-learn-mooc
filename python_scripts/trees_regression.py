# %% [markdown]
# # Decision tree for regression
#
# In this notebook, we present how decision trees are working in regression
# problems. We show differences with the decision trees previously presented in
# a classification setting.
#
# First, we will load the regression dataset presented at the beginning of this
# chapter.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

X_train, y_train = data[data_columns], data[target_column]

# %% [markdown]
# To illustrate how decision trees are predicting in a regression setting, we
# will create a synthetic dataset containing all possible flipper length from
# the minimum to the maximum of the original data.

# %%
import numpy as np

X_test = pd.DataFrame(np.arange(X_train[data_columns[0]].min(),
                                X_train[data_columns[0]].max()),
                      columns=data_columns)

# %%
import seaborn as sns
sns.set_context("talk")

_ = sns.scatterplot(data=data, x="Flipper Length (mm)", y="Body Mass (g)")

# %% [markdown]
# We will first illustrate the difference between a linear model and a decision
# tree.

# %%
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

# %%
import matplotlib.pyplot as plt

ax = sns.scatterplot(
    data=data, x="Flipper Length (mm)", y="Body Mass (g)",
    color="black", alpha=0.5)
ax.plot(X_test, y_pred, linewidth=4, label="Linear regression")
_ = plt.legend()

# %% [markdown]
# On the plot above, we see that a non-regularized `LinearRegression` is able
# to fit the data. A feature of this model is that all new predictions
# will be on the line.

# %%
ax = sns.scatterplot(
    data=data, x="Flipper Length (mm)", y="Body Mass (g)",
    color="black", alpha=0.5)
ax.plot(X_test, y_pred, linewidth=4, label="Linear regression")
ax.plot(X_test[::3], y_pred[::3], label="Test predictions",
        color="tab:orange", marker=".", markersize=15, linestyle="")
_ = plt.legend()

# %% [markdown]
# Contrary to linear models, decision trees are non-parametric models, so they
# do not make assumptions about the way data are distributed. This will affect
# the prediction scheme. Repeating the above experiment will highlight the
# differences.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=1)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# %%
ax = sns.scatterplot(
    data=data, x="Flipper Length (mm)", y="Body Mass (g)",
    color="black", alpha=0.5)
ax.plot(X_test, y_pred, linewidth=4, label="Decision tree")
_ = plt.legend()

# %% [markdown]
# We see that the decision tree model does not have a priori distribution for
# the data and we do not end-up with a straight line to regress flipper length
# and body mass.
#
# Instead, we observe that the predictions of the tree are piecewise constant.
# Indeed, our feature space was split into two partitions. We can check the
# tree structure to see what was the threshold found during the training.

# %%
from sklearn.tree import plot_tree

_ = plot_tree(tree, feature_names=data_columns)

# %% [markdown]
# The threshold for our feature (flipper length) is 202.5 mm. The predicted
# values on each side of the split are two constants: 3683.50 g and 5023.62 g.
# These values corresponds to the mean values of the training samples in each
# partition.
#
# In classification, we saw that increasing the depth of the tree allowed to
# get more complex decision boundary. We can check the effect of increasing the
# depth for decision tree in a regression setting.

# %%
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# %%
ax = sns.scatterplot(
    data=data, x="Flipper Length (mm)", y="Body Mass (g)",
    color="black", alpha=0.5)
ax.plot(X_test, y_pred, linewidth=4, label="Decision tree")
_ = plt.legend()

# %% [markdown]
# Increasing the depth of the tree will increase the number of partition and
# thus the number of constant values that the tree is capable of predicting.
#
# In this notebook, we highlighted the between decision tree in classification
# and in regression.
