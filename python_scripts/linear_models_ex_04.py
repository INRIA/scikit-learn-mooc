# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M4.04
#
# In the previous notebook, we saw the effect of applying some regularization
# on the coefficient of a linear model.
#
# In this exercise, we will study the advantage of using some regularization
# when dealing with correlated features.
#
# We will first create a regression dataset. This dataset will contain 2,000
# samples and 5 features from which only 2 features will be informative.

# %%
from sklearn.datasets import make_regression

data, target, coef = make_regression(
    n_samples=2_000,
    n_features=5,
    n_informative=2,
    shuffle=False,
    coef=True,
    random_state=0,
    noise=30,
)

# %% [markdown]
# When creating the dataset, `make_regression` returns the true coefficient
# used to generate the dataset. Let's plot this information.

# %%
import pandas as pd

feature_names = [f"Features {i}" for i in range(data.shape[1])]
coef = pd.Series(coef, index=feature_names)
coef.plot.barh()
coef

# %% [markdown]
# Create a `LinearRegression` regressor and fit on the entire dataset and
# check the value of the coefficients. Are the coefficients of the linear
# regressor close to the coefficients used to generate the dataset?

# %%
# Write your code here.

# %% [markdown]
# Now, create a new dataset that will be the same as `data` with 4 additional
# columns that will repeat twice features 0 and 1. This procedure will create
# perfectly correlated features.

# %%
# Write your code here.

# %% [markdown]
# Fit again the linear regressor on this new dataset and check the
# coefficients. What do you observe?

# %%
# Write your code here.

# %% [markdown]
# Create a ridge regressor and fit on the same dataset. Check the coefficients.
# What do you observe?

# %%
# Write your code here.

# %% [markdown]
# Can you find the relationship between the ridge coefficients and the original
# coefficients?

# %%
# Write your code here.
