# %% [markdown]
# # 📝 Exercise 01
#
# The aim of this exercise is to make the following experiments:
#
# * train and test a support vector machine classifier through
#   cross-validation;
# * study the effect of the parameter gamma of this classifier using a
#   validation curve;
# * study if it would be useful in term of classification if we could add new
#   samples in the dataset using a learning curve.
#
# To make these experiments we will first load the blood transfusion dataset.

# %%
import pandas as pd

data = pd.read_csv("../datasets/blood_transfusion.csv")
X, y = data.drop(columns="Class"), data["Class"]

# %% [markdown]
# Create a machine learning pipeline which will standardize the data and then
# use a support vector machine with an RBF kernel

# %%
# Write your code here.

# %% [markdown]
# Evaluate the performance of your model by cross-validation with a
# `ShuffleSplit` scheme.

# %%
# Write your code here.

# %% [markdown]
# The parameter `gamma` is one of the parameter controlling under/over-fitting
# in support vector machine with an RBF kernel.
# \
# Compute the validation curve to evaluate the effect of the parameter `gamma`.
# \
# You can vary its value between `10e-3` and `10e2` by generating samples on a
# logarithmic scale.

# %%
# Write your code here.

# %% [markdown]
# Plot the validation curve for the train and test scores.

# %%
# Write your code here.

# %% [markdown]
# Now, you can perform an analysis to check whether adding new samples to the
# dataset could help our model to better generalize.
# \
# Compute the learning curve by computing the train and test scores for
# different training dataset size.
# Plot the train and test scores with respect to the number of samples.

# %%
# Write your code here.
