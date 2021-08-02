# -*- coding: utf-8 -*-
# %% [markdown]
# # 📝 Exercise M6.03
#
# This exercise aims at verifying if AdaBoost can over-fit.
# We will make a grid-search and check the scores by varying the
# number of estimators.
#
# We will first load the California housing dataset and split it into a
# training and a testing set.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# Then, create an `AdaBoostRegressor` instance. Use the function
# `sklearn.model_selection.validation_curve` to get training and test scores
# by varying the number of estimators. Use the mean absolute error as a metric
# by passing `scoring="neg_mean_absolute_error"`.
# *Hint: vary the number of estimators between 1 and 60.*

# %%
# Write your code here.

# %% [markdown]
# Plot both the mean training and test errors. You can also plot the
# standard deviation of the errors.
# *Hint: you can use `plt.errorbar`.*

# %%
# Write your code here.

# %% [markdown]
# Repeat the experiment using a random forest instead of an AdaBoost regressor.

# %%
# Write your code here.
