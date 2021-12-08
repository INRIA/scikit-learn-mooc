# %% [markdown]
# # Comparing results with baseline and chance level
#
# In this notebook, we present how to compare the generalization performance of
# a model to a minimal baseline.
#
# Indeed, in the previous notebook, we compared the testing error by taking
# into account the target distribution. A good practice is to compare the
# testing error with a dummy baseline to define a chance level. In regression,
# we could use the `DummyRegressor` and predict the mean target value observed
# on the training set without using the input features.
#
# This notebook demonstrates how to compute the the score of a regression model
# and the baseline on the California housing dataset.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# section named "Appendix - Datasets description" at the end of this MOOC.
# ```

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

# %% [markdown]
# Across all evaluations, we will use a `ShuffleSplit` cross-validation
# splitter with 20% of the data held on the validation side of the split.

# %%
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

# %% [markdown]
# We start by running the cross-validation for a simple decision tree regressor
# which is our model of interest. Besides, we will store the testing error in a
# pandas series to make it easier to plot the results.

# %%
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate

regressor = DecisionTreeRegressor()
cv_results_tree_regressor = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2
)

errors_tree_regressor = pd.Series(
    -cv_results_tree_regressor["test_score"], name="Decision tree regressor"
)

# %% [markdown]
# Then, we evaluate our baseline. This baseline is called a dummy regressor.
# This dummy regressor will always predict the mean target computed on the
# training target variable. Therefore, the dummy regressor does not use any
# information from the input features stored in the dataframe named `data`.

# %%
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
result_dummy = cross_validate(
    dummy, data, target, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2
)
errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


# %% [markdown]
# We now plot the testing errors for the mean target baseline and the actual
# decision tree regressor.

# %%
all_errors = pd.concat(
    [errors_tree_regressor, errors_dummy_regressor],
    axis=1,
)

# %%
import matplotlib.pyplot as plt

all_errors.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Distribution of the testing errors")

# %% [markdown]
# We see that even if the generalization performance of our model is far from
# being perfect (price predictions are off by more than 25,000 US dollars on
# average), it is much better than the mean price baseline.
#
# Note that here we used the mean price as the baseline prediction. We could
# have used the median or an arbitrary constant instead. See the online
# documentation of the
# [sklearn.dummy.DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
# class for other options.
