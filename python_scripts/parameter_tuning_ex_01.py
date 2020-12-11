# %% [markdown]
# # 📝 Introductory example for hyperparameters tuning
#
# In this exercise, we aim at showing the effect on changing hyperparameter
# value of predictive pipeline. As an illustration, we will use a linear model
# only on the numerical features of adult census to simplify the pipeline.
#
# Let's start by loading the data.

# %%
from sklearn import set_config
set_config(display='diagram')

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

target = df[target_name]
data = df[numerical_columns]

# %% [markdown]
# We will first divide the data into a train and test set to evaluate
# the model.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# First, define a logistic regression with a preprocessing stage to scale the
# data.

# %%
# Write your code here!

# %% [markdown]
# Now, fit the model on the train set and compute the model's accuracy on the
# test set.

# %%
# Write your code here!

# %% [markdown]
# We will use this model as a baseline. Now, we will check the effect of
# changing the value of the hyperparameter `C` in logistic regression. First,
# check what is the default value of the hyperparameter `C` of the logistic
# regression.

# %%
# Write your code here!

# %% [markdown]
# Create a model by setting the `C` hyperparameter to `0.001` and compute the
# performance of the model.

# %%
# Write your code here!

# %% [markdown]
# Repeat the same experiment for `C=100`

# %%
# Write your code here!
