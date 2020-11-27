# # Comparing results with baseline and chance level
#
# Previously, we compare the generalization error by taking into account the
# target distribution. A good practice is to compare the generalization error
# with a dummy baseline and the chance level. In regression, we could use the
# `DummyRegressor` and predict the mean target without using the data. The
# chance level can be determined by permuting the labels and check the
# difference of result.

# %%
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# %%
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)

regressor = DecisionTreeRegressor()
result_regressor = cross_validate(
    regressor,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
)
errors_regressor = pd.Series(
    -result_regressor["test_score"], name="Regressor error")

# %%
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor()
result_dummy = cross_validate(
    dummy,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
)
errors_dummy = pd.Series(
    -result_dummy["test_score"], name="Dummy error")

# %%
from sklearn.model_selection import permutation_test_score

score, permutation_score, pvalue = permutation_test_score(
    regressor,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    n_permutations=30,
)
errors_permutation = pd.Series(
    -permutation_score, name="Permuted error")

# %% [markdown]
# We plot the generalization errors for each of the experiments. Even if our
# regressor does not perform well, it is far above a regressor that would
# predict the mean target.

# %%
final_errors = pd.concat(
    [errors_regressor, errors_dummy, errors_permutation],
    axis=1,
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

sns.displot(final_errors, kind="kde")
_ = plt.xlabel("Mean absolute error (k$)")
