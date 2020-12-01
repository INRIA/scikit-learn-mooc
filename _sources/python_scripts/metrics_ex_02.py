# %% [markdown]
# # 📝 Exercise 02
#
# As for the exercise for the classification metrics, in this notebook we
# intend to use the regression metrics within a cross-validation framework
# to get familiar with the syntax.
#
# We will use the Ames house prices dataset.

# %%
import pandas as pd
import numpy as np

data = pd.read_csv("../datasets/house_prices.csv")
X, y = data.drop(columns="SalePrice"), data["SalePrice"]
X = X.select_dtypes(np.number)
y /= 1000

# %% [markdown]
# The first step will be to create a linear regression model.

# %%
# Write your code here.

# %% [markdown]
# Then, use the `cross_val_score` to estimate the performance of the model.
# Use a `KFold` cross-validation with 10 folds. Make it explicit to use the
# $R^2$ score by assigning the paramater `scoring` even if it is the default
# score.

# %%
# Write your code here.

# %% [markdown]
# Then, instead of using the $R^2$ score, use the mean absolute error. You need
# to check the documentation for the `scoring` parameter.

# %%
# Write your code here.

# %% [markdown]
# Finally, use the `cross_validate` function and compute multiple score/error
# at once by passing a list to the `scoring` parameter. You can compute the
# $R^2$ score and the mean absolute error.

# %%
# Write your code here.
