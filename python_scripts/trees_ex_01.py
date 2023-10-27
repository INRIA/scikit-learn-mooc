# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M5.01
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
# Write your code here.

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
# Write your code here.

# %% [markdown]
# Did we make use of the feature "Culmen Length"? Plot the tree using the
# function `sklearn.tree.plot_tree` to find out!

# %%
# Write your code here.

# %% [markdown]
# Compute the accuracy of the decision tree on the testing data.

# %%
# Write your code here.
