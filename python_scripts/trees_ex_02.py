# %% [markdown]
# # Exercise 02
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
# TODO

# %% [markdown]
# Create a testing dataset, ranging from the minimum to the maximum of the
# flipper length of the training dataset. Get the predictions of each model
# using this test dataset.

# %%
# TODO

# %% [markdown]
# Create a scatter plot containing the training samples and superimpose the
# predictions of both model on the top.

# %%
# TODO

# %% [markdown]
# Now, we will check the extrapolation capabilities of each model. Create a
# dataset containing the value of your previous dataset. Besides add values
# below and above the minimum and the maximum of the flipper length seen
# during training.

# %%
# TODO

# %% [markdown]
# Finally, make predictions with both model on this new testing set. Repeat
# the plotting of the previous exercise.

# %%
# TODO
