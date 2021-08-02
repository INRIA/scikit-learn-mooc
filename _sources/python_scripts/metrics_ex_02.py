# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M7.03
#
# As with the classification metrics exercise, we will evaluate the regression
# metrics within a cross-validation framework to get familiar with the syntax.
#
# We will use the Ames house prices dataset.

# %%
import pandas as pd
import numpy as np

ames_housing = pd.read_csv("../datasets/house_prices.csv")
data = ames_housing.drop(columns="SalePrice")
target = ames_housing["SalePrice"]
data = data.select_dtypes(np.number)
target /= 1000

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```


# %% [markdown]
# The first step will be to create a linear regression model.

# %%
# Write your code here.

# %% [markdown]
# Then, use the `cross_val_score` to estimate the generalization performance of
# the model. Use a `KFold` cross-validation with 10 folds. Make the use of the
# $R^2$ score explicit by assigning the parameter `scoring` (even though it is
# the default score).

# %%
# Write your code here.

# %% [markdown]
# Then, instead of using the $R^2$ score, use the mean absolute error. You need
# to refer to the documentation for the `scoring` parameter.

# %%
# Write your code here.

# %% [markdown]
# Finally, use the `cross_validate` function and compute multiple scores/errors
# at once by passing a list of scorers to the `scoring` parameter. You can
# compute the $R^2$ score and the mean absolute error for instance.

# %%
# Write your code here.
