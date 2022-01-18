# %% [markdown]
# # 📝 Introductory exercise regarding stratification
#
# The goal of this exercise is to highlight one limitation of
# applying blindly a k-fold cross-validation.
#
# In this exercise we will use the iris dataset.

# %%
from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True, as_frame=True)

# %% [markdown]
# Create a decision tree classifier that we will use in the next experiments.

# %%
# Write your code here.

# %% [markdown]
# As a first experiment, use the utility
# `sklearn.model_selection.train_test_split` to split the data into a train
# and test set. Train the classifier using the train set and check the score
# on the test set.

# %%
# Write your code here.

# %% [markdown]
# Now, use the utility `sklearn.model_selection.cross_val_score` with a
# `sklearn.model_selection.KFold` by setting only `n_splits=3`. Check the
# results on each fold. Explain the results.

# %%
# Write your code here.
