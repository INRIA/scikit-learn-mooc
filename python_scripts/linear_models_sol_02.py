# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M4.02
#
# The goal of this exercise is to build an intuition on what will be the
# parameters' values of a linear model when the link between the data and the
# target is non-linear.
#
# First, we will generate such non-linear data.
#
# ```{tip}
# `np.random.RandomState` allows to create a random number generator which can
# be later used to get deterministic results.
# ```

# %%
import numpy as np
# Set the seed for reproduction
rng = np.random.RandomState(0)

# Generate data
n_sample = 100
data_max, data_min = 1.4, -1.4
len_data = (data_max - data_min)
data = rng.rand(n_sample) * len_data - len_data / 2
noise = rng.randn(n_sample) * .3
target = data ** 3 - 0.5 * data ** 2 + noise

# %% [markdown]
# ```{note}
# To ease the plotting, we will create a Pandas dataframe containing the data
# and target
# ```

# %%
import pandas as pd
full_data = pd.DataFrame({"data": data, "target": target})

# %%
import seaborn as sns

_ = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                    alpha=0.5)

# %% [markdown]
# We observe that the link between the data `data` and vector `target` is
# non-linear. For instance, `data` could represent the years of
# experience (normalized) and `target` the salary (normalized). Therefore, the
# problem here would be to infer the salary given the years of experience.
#
# Using the function `f` defined below, find both the `weight` and the
# `intercept` that you think will lead to a good linear model. Plot both the
# data and the predictions of this model.

# %%
def f(data, weight=0, intercept=0):
    target_predict = weight * data + intercept
    return target_predict

# %%
# solution
predictions = f(data, weight=1.2, intercept=-0.2)

# %% tags=["solution"]
ax = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                     alpha=0.5)
_ = ax.plot(data, predictions)

# %% [markdown]
# Compute the mean squared error for this model

# %%
# solution
from sklearn.metrics import mean_squared_error

error = mean_squared_error(target, f(data, weight=1.2, intercept=-0.2))
print(f"The MSE is {error}")

# %% [markdown]
# Train a linear regression model on this dataset.
#
# ```{warning}
# In scikit-learn, by convention `data` (also called `X` in the scikit-learn
# documentation) should be a 2D matrix of shape `(n_samples, n_features)`.
# If `data` is a 1D vector, you need to reshape it into a matrix with a
# single column if the vector represents a feature or a single row if the
# vector represents a sample.
# ```

# %%
from sklearn.linear_model import LinearRegression

# solution
linear_regression = LinearRegression()
data_2d = data.reshape(-1, 1)
linear_regression.fit(data_2d, target)

# %% [markdown]
# Compute predictions from the linear regression model and plot both the data
# and the predictions.

# %%
# solution
predictions = linear_regression.predict(data_2d)

# %% tags=["solution"]
ax = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                     alpha=0.5)
_ = ax.plot(data, predictions)

# %% [markdown]
# Compute the mean squared error

# %%
# solution
error = mean_squared_error(target, predictions)
print(f"The MSE is {error}")
