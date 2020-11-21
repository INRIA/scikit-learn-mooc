# %% [markdown]
# # Solution for Exercise 02
#
# The aim of this exercise is to find out whether or not a model is able to
# extrapolate.
#
# By extrapolation, we refer to values predicted by a model outside of the
# range of feature values seen during the training.
#
# We will first load the regression data.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

X_train, y_train = data[data_columns], data[target_column]

# %% [markdow]
# First, create two models, a linear regression model and a decision tree
# regression model, and fit them on the training data. Limit the depth at
# 3 levels for the decision tree.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

linear_regression = LinearRegression()
tree = DecisionTreeRegressor(max_depth=3)

linear_regression.fit(X_train, y_train)
tree.fit(X_train, y_train)

# %% [markdown]
# Create a testing dataset, ranging from the minimum to the maximum of the
# flipper length of the training dataset. Get the predictions of each model
# using this test dataset.

# %%
import numpy as np

X_test = pd.DataFrame(
    np.arange(X_train[data_columns[0]].min(), X_train[data_columns[0]].max()),
    columns=data_columns,
)

# %%
y_pred_linear_regression = linear_regression.predict(X_test)
y_pred_tree = tree.predict(X_test)

# %% [markdown]
# Create a scatter plot containing the training samples and superimpose the
# predictions of both model on the top.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.scatterplot(
    data=data,
    x="Flipper Length (mm)",
    y="Body Mass (g)",
    color="black",
    alpha=0.5,
)
ax.plot(
    X_test, y_pred_linear_regression, linewidth=4, label="Linear regression"
)
ax.plot(X_test, y_pred_tree, linewidth=4, label="Decision tree regression")
_ = plt.legend()

# %% [markdown]
# The predictions that we got where within the range of feature values seen
# during training. In some sense, we observe the capabilities of our model to
# interpolate.
#
# Now, we will check the extrapolation capabilities of each model. Create a
# dataset containing the value of your previous dataset. Besides add values
# below and above the minimum and the maximum of the flipper length seen
# during training.

# %%
offset = 30
X_test = pd.DataFrame(
    np.arange(
        X_train[data_columns[0]].min() - offset,
        X_train[data_columns[0]].max() + offset,
    ),
    columns=data_columns,
)

# %% [markdown]
# Finally, make predictions with both model on this new testing set. Repeat
# the plotting of the previous exercise.

# %%
y_pred_linear_regression = linear_regression.predict(X_test)
y_pred_tree = tree.predict(X_test)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.scatterplot(
    data=data,
    x="Flipper Length (mm)",
    y="Body Mass (g)",
    color="black",
    alpha=0.5,
)
ax.plot(
    X_test, y_pred_linear_regression, linewidth=4, label="Linear regression"
)
ax.plot(X_test, y_pred_tree, linewidth=4, label="Decision tree regression")
_ = plt.legend()

# %% [markdown]
# The linear model will extrapolate using the fitted model for flipper lengths
# < 175 mm and > 235 mm. In fact, we are using the model parametrization to
# make this predictions.
#
# As mentioned, decision trees are non-parametric models and we observe that
# they cannot extrapolate. For flipper lengths below the minimum, the mass of
# the penguin in the training data with the shortest flipper length will always
# be predicted. Similarly, for flipper lengths above the maximum, the mass of
# the penguin in the training data with the longest flipper will always
# predicted.
