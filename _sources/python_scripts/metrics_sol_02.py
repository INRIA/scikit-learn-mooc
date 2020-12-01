# %% [markdown]
# # 📃 Solution for Exercise 02
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
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# %% [markdown]
# Then, use the `cross_val_score` to estimate the performance of the model.
# Use a `KFold` cross-validation with 10 folds. Make it explicit to use the
# $R^2$ score by assigning the paramater `scoring` even if it is the default
# score.

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=10, scoring="r2")
print(f"R2 score: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# Then, instead of using the $R^2$ score, use the mean absolute error. You need
# to check the documentation for the `scoring` parameter.

# %%
scores = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error")
errors = -scores
print(f"Mean absolute error: "
      f"{errors.mean():.3f} k$ +/- {errors.std():.3f}")

# %% [markdown]
# The `scoring` parameter in scikit-learn expects score. It means that higher
# values means a better model. However, errors are the opposite: the smaller,
# the better. Therefore, the error should be multiply by -1. That's why, the
# string given the `scoring` starts with `neg_` when dealing with metrics which
# are errors.
#
# Finally, use the `cross_validate` function and compute multiple score/error
# at once by passing a list to the `scoring` parameter. You can compute the
# $R^2$ score and the mean absolute error.

# %%
from sklearn.model_selection import cross_validate

scoring = ["r2", "neg_mean_absolute_error"]
cv_results = cross_validate(model, X, y, scoring=scoring)

# %%
import pandas as pd

scores = {"R2": cv_results["test_r2"],
          "MSE": -cv_results["test_neg_mean_absolute_error"]}
scores = pd.DataFrame(scores)
scores
