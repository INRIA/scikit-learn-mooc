# %% [markdown]
# # 📝 Introductory exercise for sample grouping
#
# This exercise aims at highlighting issues that one could encounter when
# discarding grouping pattern existing in a dataset.
#
# We will use the digits dataset which includes some grouping pattern.

# %%
from sklearn.datasets import load_digits

data, target = load_digits(return_X_y=True, as_frame=True)

# %% [markdown]
# The first step is to create a model. Use a machine learning pipeline
# composed of a scaler followed by a logistic regression classifier.

# %%
# Write your code here.

# %% [markdown]
# Then, create a a `KFold` object making sure that the data will not be
# shuffled during the cross-validation. Use the previous model, data, and
# cross-validation strategy defined to estimate the generalization performance of
# the model.

# %%
# Write your code here.

# %% [markdown]
# Finally, perform the same experiment by shuffling the data within the
# cross-validation. Draw some conclusion regarding the dataset.

# %%
# Write your code here.
