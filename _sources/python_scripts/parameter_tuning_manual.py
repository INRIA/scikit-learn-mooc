# %% [markdown]
# # Set and get hyperparameters in scikit-learn
#
# This notebook shows how one can get and set the value of hyperparameter in
# a scikit-learn estimator.
#
# We will start by loading the adult census dataset and only use the numerical
# feature. Subsequently, we will divide the dataset into two subsets.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

target = df[target_name]
data = df[numerical_columns]

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# Let's create a simple predictive model made of a scaling followed by a
# logistic regression classifier.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression())
])
model.fit(data_train, target_train)

# %%
model.score(data_test, target_test)

# %% [markdown]
# We created a model with default `C` value that is equal to 1. We saw in the
# previous exercise that we will be interested to set the value of an
# hyperparameter. One possibility is to set the parameter when we create the
# model instance. However, we might be interested to set the value of the
# parameter after the instance is created.
#
# Actually scikit-learn estimators have a `set_params` method that allows you
# to change the parameter of a model after it has been created. For example, we
# can set `C=1e-3` and fit and evaluate the model:

# %%
model.set_params(classifier__C=1e-3)
model.fit(data_train, target_train)
print(f"The test accuracy score of the model is: "
      f"{model.score(data_test, target_test):.3f}")

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
# In this notebook we have seen:
#
# - how to use `get_params` and `set_params` to get the parameters of a model
#   and set them.
