# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M5.01
#
# In the previous notebook, we showed how a tree with a depth of 1 level was
# working. The aim of this exercise is to repeat part of the previous
# experiment for a depth with 2 levels to show how the process of partitioning
# is repeated over time.
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
# Create a decision tree classifier with a maximum depth of 2 levels and fit
# the training data. Once this classifier trained, plot the data and the
# decision boundary to see the benefit of increasing the depth. To plot the
# decision boundary, you should import the class `DecisionBoundaryDisplay`
# from the module `sklearn.inspection` as shown in the previous course notebook.

# %%
# Write your code here.

# %% [markdown]
# Did we make use of the feature "Culmen Length"?
# Plot the tree using the function `sklearn.tree.plot_tree` to find out!

# %%
# Write your code here.

# %% [markdown]
# Compute the accuracy of the decision tree on the testing data.

# %%
# Write your code here.
