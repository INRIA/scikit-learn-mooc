# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M4.03
#
# In all previous notebooks, we only used a single feature in `data`. But we
# have already shown that we could add new features to make the model more
# expressive by deriving new features, based on the original feature.
#
# The aim of this notebook is to train a linear regression algorithm on a
# dataset with more than a single feature.
#
# We will load a dataset about house prices in California.
# The dataset consists of 8 features regarding the demography and geography of
# districts in California and the aim is to predict the median house price of
# each district. We will use all 8 features to predict the target, the median
# house price.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
data.head()

# %% [markdown]
# Now it is your turn to train a linear regression model on this dataset.
# First, create a linear regression model.

# %%
# solution
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# %% [markdown]
# Execute a cross-validation with 10 folds and use the mean absolute error
# (MAE) as metric. Be sure to *return* the fitted *estimators*.

# %%
# solution
from sklearn.model_selection import cross_validate

cv_results = cross_validate(linear_regression, data, target,
                            scoring="neg_mean_absolute_error",
                            return_estimator=True, cv=10, n_jobs=2)

# %% [markdown]
# Compute the mean and std of the MAE in thousands of dollars (k$).

# %%
# solution
print(f"Mean absolute error on testing set: "
      f"{-cv_results['test_score'].mean():.3f} k$ ± "
      f"{cv_results['test_score'].std():.3f}")

# %% [markdown]
# Inspect the fitted model using a box plot to show the distribution of values
# for the coefficients returned from the cross-validation. Hint:
# use the function
# [`df.plot.box()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.box.html)
# to create a box plot.


# %%
# solution
import pandas as pd

weights = pd.DataFrame(
    [est.coef_ for est in cv_results["estimator"]], columns=data.columns)

# %% tags=["solution"]
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
weights.plot.box(color=color, vert=False)
_ = plt.title("Value of linear regression coefficients")
