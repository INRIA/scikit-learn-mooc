# %% [markdown]
# # Exercise 02
#
# The goal of this exercise is to build an intuition on what will be the
# parameters' values of a linear model when the link between the data and the
# target is non-linear.
#
# First, we will generate such non-linear data.


# %%
import numpy as np
# fix the seed for reproduction
rng = np.random.RandomState(0)

# generate data
n_sample = 100
x_max, x_min = 1.4, -1.4
len_x = (x_max - x_min)
x = rng.rand(n_sample) * len_x - len_x / 2
noise = rng.randn(n_sample) * .3
y = x ** 3 - 0.5 * x ** 2 + noise

# %%
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.xlabel('x', size=26)
_ = plt.ylabel('y', size=26)

# %% [markdown]
# We observe that the link between the data `x` and target `y` is non-linear.
# For instance, x could represent to be the years of experience (normalized)
# and y the salary (normalized). Therefore, the problem here would be to infer
# the salary given the years of experience.
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
# TODO: plot both the data and the model predictions

# %%
# TODO: compute the mean squared error

# %% [markdown]
# Train a linear regression model and plot both the data and the predictions
# of the model. Compute the mean squared error with this model.

# %%
from sklearn.linear_model import LinearRegression

# TODO: fit a linear regression

# %%
# TODO: plot the data and the prediction of the linear regression model

# %%
# TODO: compute the mean squared error
