# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to hyper-parameter tuning
#
# The process of learning a predictive model is driven by a set of internal
# parameters and a set of training data. These internal parameters are called
# hyper-parameters and are specific for each family of models. In addition, a
# specific set of hyper-parameters are optimal for a specific dataset and thus
# they need to be optimized. In this notebook we will use the words
# "hyper-parameters" and "parameters" interchangeably.
#
# This notebook shows the influence of changing model hyper-parameters.

# %% [markdown]
# We will reload the adult census dataset and ignore some of the columns
# as previously done in previous notebooks.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

# %%
target_name = "class"
target = df[target_name]
target

# %%
data = df.drop(columns=[target_name, "fnlwgt", "education-num"])
data.head()

# %% [markdown]
# Once the dataset is loaded, we split it into a training and testing sets.

# %%
from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# Then, we define the preprocessing pipeline to transform differently
# the numerical and categorical data, identically to the previous notebook.
# We will use an ordinal encoder for the categories because we will use an
# histogram gradient-boosting as predictive model.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

from sklearn.preprocessing import OrdinalEncoder

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

categories = [
    data[column].unique() for column in data[categorical_columns]]

categorical_preprocessor = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer([
    ('cat-preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# %% [markdown]
# Finally, we use a tree-based classifier (i.e. histogram gradient-boosting) to
# predict whether or not a person earns more than 50,000 dollars a year.

# %%
# %%time
# for the moment this line is required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42))])
model.fit(df_train, target_train)

print(
    f"The test accuracy score of the gradient boosting pipeline is: "
    f"{model.score(df_test, target_test):.2f}")

# In the previous example, we created an histogram gradient-boosting classifier
# using the default parameters by omitting to explicitely set these parameters.
#
# However, there is no reason that these parameters are optimal for our
# dataset. For instance, we can try to set the `learning_rate` parameter and
# see how it changes the score:

# %%
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42, learning_rate=1e-3))
])
model.fit(df_train, target_train)
print(
    f"The test accuracy score of the gradient boosting pipeline is: "
    f"{model.score(df_test, target_test):.2f}")

# %%
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42, learning_rate=10))
])
model.fit(df_train, target_train)
print(
    f"The test accuracy score of the gradient boosting pipeline is: "
    f"{model.score(df_test, target_test):.2f}")

# # %% [markdown]
# ## Quizz
#
# 1. What is the default value of the `learning_rate` parameter of the
# `HistGradientBoostingClassifier` class? ([link to the API documentation](
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html))
#
# 2. Decrease progressively value of `learning_rate`: can you find a value that
# yields an accuracy higher than with the default learning rate?
#
# 3. Fix `learning_rate` to 0.05 and try setting the value of `max_leaf_nodes`
# to the minimum value of 2. Does it improve the accuracy?
#
# 4. Try to progressively increase the value of `max_leaf_nodes` to 256 by
# taking powers of 2. What do you observe?

# %% [markdown]
# Actually scikit-learn estimators have a `set_params` method that allows you
# to change the parameter of a model after it has been created. For example, we
# can set the `learning_rate=1e-3` and check that we get the same score as
# previously:

# %%
model.set_params(classifier__learning_rate=1e-3)
model.fit(df_train, target_train)
print(
    f"The test accuracy score of the gradient boosting pipeline is: "
    f"{model.score(df_test, target_test):.2f}")


# %% [markdown]
# When the model of interest is a `Pipeline`, the parameter names are of the
# form `<model_name>__<parameter_name>` (note the double underscore in the
# middle). In our case, `classifier` comes from the `Pipeline` definition and
# `learning_rate` is the parameter name of `HistGradientBoostingClassifier`.
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
# parameter, for example `classifier__learning_rate`, you can use:

# %%
model.get_params()['classifier__learning_rate']

# %% [markdown]
# In this notebook we have seen:
#
# - how hyper-parameters can affect the performance of a model;
# - how to use `get_params` and `set_params` to get the parameters of a model
#   and set them.
