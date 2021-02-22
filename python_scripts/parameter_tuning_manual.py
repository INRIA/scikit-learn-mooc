# %% [markdown]
# # Set and get hyperparameters in scikit-learn
#
# This notebook shows how one can get and set the value of hyperparameter in
# a scikit-learn estimator. We recall that hyperparameters refer to the
# parameter that will control the learning process.
#
# They should not be confused with the fitted parameters, resulting from the
# training. These fitted parameters are recognizable in scikit-learn because
# they are spelled with a final underscore `_`, for instance `model.coef_`.
#
# We will start by loading the adult census dataset and only use the numerical
# feature.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

target = adult_census[target_name]
data = adult_census[numerical_columns]

# %% [markdown]
# ```{caution}
# Here and later, we use the name `data` and `target` to be explicit. In
# scikit-learn, documentation `data` is commonly named `X` and `target` is
# commonly called `y`.
# ```

# %% [markdown]
# Our data is only numerical.

# %%
data.head()

# %% [markdown]
# Let's create a simple predictive model made of a scaler followed by a
# logistic regression classifier.
#
# As mentioned in previous notebooks, many models, including linear ones,
# work better if all features have a similar scaling. For this purpose,
# we use a `StandardScaler`, which transforms the data by rescaling features.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression())
])

# %% [markdown]
# We can evaluate the statistical performance of the model via
# cross-validation.

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target)
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# We created a model with the default `C` value that is equal to 1. We saw in
# the previous exercise that we will be interested to set the value of an
# hyperparameter. One possibility is to set the parameter when we create the
# model instance. However, we might be interested to set the value of the
# parameter after the instance is created.
#
# Actually scikit-learn estimators have a `set_params` method that allows you
# to change the parameter of a model after it has been created. For example, we
# can set `C=1e-3` and fit and evaluate the model:

# %%
model.set_params(classifier__C=1e-3)
cv_results = cross_validate(model, data, target)
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# When the model of interest is a `Pipeline`, the parameter names are of the
# form `<model_name>__<parameter_name>` (note the double underscore in the
# middle). In our case, `classifier` comes from the `Pipeline` definition and
# `C` is the parameter name of `LogisticRegression`.
#
# In general, you can use the `get_params` method on scikit-learn models to
# list all the parameters with their values. For example, if you want to
# get all the parameter names, you can use:

# %%
for parameter in model.get_params():
    print(parameter)

# %% [markdown]
# `.get_params()` returns a `dict` whose keys are the parameter names and whose
# values are the parameter values. If you want to get the value of a single
# parameter, for example `classifier__C`, you can use:

# %%
model.get_params()['classifier__C']

# %% [markdown]
# We can vary systematically the value of C to see if there is an optimal
# value.

# %%
for C in [1e-3, 1e-2, 1e-1, 1, 10]:
    model.set_params(classifier__C=C)
    cv_results = cross_validate(model, data, target)
    scores = cv_results["test_score"]
    print(f"Accuracy score via cross-validation with C={C}:\n"
          f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# We can see that as long as C is high enough, the model seems to perform
# well.
#
# What we did here is very manual: it involves scanning the values for C
# and picking the best one manually. In the next lesson, we will see how
# to do this automatically.
#
# ```{warning}
# When we evaluate a family of models on test data and pick the best
# performer, we can not trust the corresponding prediction accuracy, and
# we need to apply the selected model to new data. Indeed, the test data
# has been used to select the model, and it is thus no longer independent
# from this model.
# ```

# %% [markdown]
# In this notebook we have seen:
#
# - how to use `get_params` and `set_params` to get the parameters of a model
#   and set them.
