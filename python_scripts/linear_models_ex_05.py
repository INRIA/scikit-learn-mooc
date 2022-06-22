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
# # 📝 Exercise M4.05
# In the previous notebook we set `penalty="none"` to disable regularization
# entirely. This parameter can also control the **type** of regularization to use,
# whereas the regularization **strength** is set using the parameter `C`.
# Setting`penalty="none"` is equivalent to an infinitely large value of `C`.
# In this exercise, we ask you to train a logistic regression classifier using the
# `penalty="l2"` regularization (which happens to be the default in scikit-learn)
# to find by yourself the effect of the parameter `C`.
#
# We will start by loading the dataset.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")
# only keep the Adelie and Chinstrap classes
penguins = penguins.set_index("Species").loc[
    ["Adelie", "Chinstrap"]].reset_index()

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %%
from sklearn.model_selection import train_test_split

penguins_train, penguins_test = train_test_split(penguins, random_state=0)

data_train = penguins_train[culmen_columns]
data_test = penguins_test[culmen_columns]

target_train = penguins_train[target_column]
target_test = penguins_test[target_column]

# %% [markdown]
# First, let's create our predictive model.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2"))

# %% [markdown]
# Given the following candidates for the `C` parameter, find out the impact of
# `C` on the classifier decision boundary. You can use
# `sklearn.inspection.DecisionBoundaryDisplay.from_estimator` to plot the
# decision function boundary.

# %%
Cs = [0.01, 0.1, 1, 10]

# Write your code here.

# %% [markdown]
# Look at the impact of the `C` hyperparameter on the magnitude of the weights.

# %%
# Write your code here.
