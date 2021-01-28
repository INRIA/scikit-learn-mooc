# %% [markdown]
# # 📃 Solution for Exercise 02
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
x_max, x_min = 1.4, -1.4
len_x = (x_max - x_min)
x = rng.rand(n_sample) * len_x - len_x / 2
noise = rng.randn(n_sample) * .3
y = x ** 3 - 0.5 * x ** 2 + noise

# %% [markdown]
# ```{note}
# To ease the plotting, we will create a Pandas dataframe containing the data
# and target
# ```

# %%
import pandas as pd
data = pd.DataFrame({"x": x, "y": y})

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

_ = sns.scatterplot(data=data, x="x", y="y")

# %% [markdown]
# We observe that the link between the data `x` and target `y` is non-linear.
# For instance, `x` could represent to be the years of experience (normalized)
# and `y` the salary (normalized). Therefore, the problem here would be to
# infer the salary given the years of experience.
#
# Using the function `f` defined below, find both the `weight` and the
# `intercept` that you think will lead to a good linear model. Plot both the
# data and the predictions of this model. Compute the mean squared error as
# well.


# %%
def f(x, weight=0, intercept=0):
    y_predict = weight * x + intercept
    return y_predict


# %%
ax = sns.scatterplot(data=data, x="x", y="y")
_ = ax.plot(x, f(x, weight=1.2, intercept=-0.2), color="tab:orange")

# %%
from sklearn.metrics import mean_squared_error

error = mean_squared_error(y, f(x, weight=1.2, intercept=0.2))
print(f"The MSE is {error}")

# %% [markdown]
# Train a linear regression model and plot both the data and the predictions
# of the model. Compute the mean squared error with this model.

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
X = x.reshape(-1, 1)
linear_regression.fit(X, y)

ax = sns.scatterplot(data=data, x="x", y="y")
_ = ax.plot(x, linear_regression.predict(X), color="tab:orange")

# %% [markdown]
# ```{warning}
# In scikit-learn, by convention `X` should be a 2D matrix of shape
# `(n_samples, n_features)`. If `X` is a 1D vector, you need to reshape it
# into a matrix with a single column if the vector represents a feature or a
# single row if the vector represents a sample.
# ```

# %%
error = mean_squared_error(y, linear_regression.predict(X))
print(f"The MSE is {error}")
