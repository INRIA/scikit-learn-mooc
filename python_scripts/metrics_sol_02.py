# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M7.03
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
# solution
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# %% [markdown]
# Then, use the `cross_val_score` to estimate the generalization performance of
# the model. Use a `KFold` cross-validation with 10 folds. Make the use of the
# $R^2$ score explicit by assigning the parameter `scoring` (even though it is
# the default score).

# %%
# solution
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data, target, cv=10, scoring="r2")
print(f"R2 score: {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# Then, instead of using the $R^2$ score, use the mean absolute error. You need
# to refer to the documentation for the `scoring` parameter.

# %%
# solution
scores = cross_val_score(model, data, target, cv=10,
                         scoring="neg_mean_absolute_error")
errors = -scores
print(f"Mean absolute error: "
      f"{errors.mean():.3f} k$ ± {errors.std():.3f}")

# %% [markdown] tags=["solution"]
# The `scoring` parameter in scikit-learn expects score. It means that the
# higher the values, and the smaller the errors are, the better the model is.
# Therefore, the error should be multiplied by -1. That's why the string given
# the `scoring` starts with `neg_` when dealing with metrics which are errors.

# %% [markdown] 
# Finally, use the `cross_validate` function and compute multiple scores/errors
# at once by passing a list of scorers to the `scoring` parameter. You can
# compute the $R^2$ score and the mean absolute error for instance.

# %%
# solution
from sklearn.model_selection import cross_validate

scoring = ["r2", "neg_mean_absolute_error"]
cv_results = cross_validate(model, data, target, scoring=scoring)

# %% tags=["solution"]
import pandas as pd

scores = {"R2": cv_results["test_r2"],
          "MAE": -cv_results["test_neg_mean_absolute_error"]}
scores = pd.DataFrame(scores)
scores
