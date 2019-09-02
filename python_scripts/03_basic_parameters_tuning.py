# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to scikit-learn: basic model hyper-parameters tuning
#
# The process to learn a predictive model is driven by a set of internal
# parameters and a set of training data. These internal parameters are called
# hyper-parameters and are specific for each family of models. In addition,
# a set of parameters are optimal for a specific dataset and thus they need
# to be optimized.
#
# This notebook shows:
# * the influence of changing model parameters;
# * how to tune these hyper-parameters;
# * how to evaluate the model performance together with hyper-parameters
#   tuning.

# %%
import pandas as pd

df = pd.read_csv("https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

# %%
target_name = "class"
target = df[target_name].to_numpy()
target

# %%
data = df.drop(columns=[target_name, "fnlwgt"])
data.head()

# %% [markdown]
# Once the dataset is loaded, we split it into a training and testing sets.

# %%
from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
# Then, we define the preprocessing pipeline to transform differently
# the numerical and categorical data.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

binary_encoding_columns = ['sex']
one_hot_encoding_columns = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race',
                            'native-country']
scaling_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'education-num']

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])

# %% [markdown]
# Finally, we use a linear classifier (i.e. logistic regression) to predict
# whether or not a person earn more than 50,000 dollars a year.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    preprocessor, LogisticRegression(max_iter=1000, solver='lbfgs')
)
model.fit(df_train, target_train)
print(
    f"The accuracy score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f}"
)

# %% [markdown]
# ## The issue of finding the best model parameters
#
# In the previous example, we created a `LogisticRegression` classifier using
# the default parameters by omitting setting explicitly these parameters.
#
# For this classifier, the parameter `C` governes the penalty; in other
# words, how much our model should "trust" (or fit) the training data.
#
# Therefore, the default value of `C` is never certified to give the best
# performing model.
#
# We can make a quick experiment by changing the value of `C` and see the
# impact of this parameter on the model performance.

# %%
C = 1
model = make_pipeline(
    preprocessor, LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
)
model.fit(df_train, target_train)
print(
    f"The accuracy score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with C={C}"
)

# %%
C = 1e-5
model = make_pipeline(
    preprocessor, LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
)
model.fit(df_train, target_train)
print(
    f"The accuracy score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with C={C}"
)

# %% [markdown]
# ## Finding the best model hyper-parameters via exhaustive parameters search
#
# We see that the parameter `C` as a significative impact on the model
# performance. This parameter should be tuned to get the best cross-validation
# score, so as to avoid over-fitting problems.
#
# In short, we will set the parameter, train our model on some data, and
# evaluate the model performance on some left out data. Ideally, we will select
# the parameter leading to the optimal performance on the testing set.
# Scikit-learn provides a `GridSearchCV` estimator which will handle the
# cross-validation and hyper-parameter search for us.

# %%
from sklearn.model_selection import GridSearchCV

model = make_pipeline(
    preprocessor, LogisticRegression(max_iter=1000, solver='lbfgs')
)

# %% [markdown]
# We will see that we need to provide the name of the parameter to be set.
# Thus, we can use the method `get_params()` to have the list of the parameters
# of the model which can set during the grid-search.

# %%
print("The hyper-parameters are for a logistic regression model are:")
for param_name in LogisticRegression().get_params().keys():
    print(param_name)

# %%
print("The hyper-parameters are for the full-pipeline are:")
for param_name in model.get_params().keys():
    print(param_name)

# %% [markdown]
# The parameter `'logisticregression__C'` is the parameter for which we would
# like different values. Let see how to use the `GridSearchCV` estimator for
# doing such search.

# %%
import time
import numpy as np

param_grid = {'logisticregression__C': (0.1, 1.0, 10.0)}
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=4, cv=5)
start = time.time()
model_grid_search.fit(df_train, target_train)
elapsed_time = time.time() - start
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f} in "
    f"{elapsed_time:.3f} seconds"
)

# %% [markdown]
# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all possible parameters combination. Once the grid-search fitted, it can be
# used as any other predictor by calling `predict` and `predict_proba`.
# Internally, it will use the model with the best parameters found during
# `fit`. You can know about these parameters by looking at the `best_params_`
# attribute.

# %%
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %% [markdown]
# With the `GridSearchCV` estimator, the parameters need to be specified
# explicitely. Instead, one could randomly generate (following a specific
# distribution) the parameter candidates. The `RandomSearchCV` allows for such
# stochastic search. It is used similarly to the `GridSearchCV` but the
# sampling distributions need to be specified instead of the parameter values.

# %%
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'logisticregression__C': uniform(loc=50, scale=100)}
model_grid_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=3, n_jobs=4, cv=5
)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %% [markdown]
# ## Notes on search efficiency
#
# Be aware that sometimes, scikit-learn provides some `EstimatorCV` classes
# which will perform internally the cross-validation in such way that it will
# more computationally efficient. We can give the example of the
# `LogisticRegressionCV` which can be used to find the best `C` in a more
# efficient way than what we previously did with the `GridSearchCV`.

# %%
from sklearn.linear_model import LogisticRegressionCV

# define the different Cs to try out
param_grid = {"C": (0.1, 1.0, 10.0)}

model = make_pipeline(
    preprocessor,
    LogisticRegressionCV(Cs=param_grid['C'], max_iter=1000, solver='lbfgs',
                         n_jobs=4, cv=5)
)
start = time.time()
model.fit(df_train, target_train)
elapsed_time = time.time() - start
print(f"Time elapsed to train LogisticRegressionCV: "
      f"{elapsed_time:.3f} seconds")

# %% [markdown]
# The `fit` time for the `CV` version of `LogisticRegression` give a speed-up
# x2. This speed-up is provided by re-using the values of coefficients to
# warm-start the estimator for the different `C` values.

# %% [markdown]
# ## Exercises:
#
# - Build a machine learning pipeline:
#       * preprocess the categorical columns using an `OrdinalEncoder` and let
#         the numerical columns as they are.
#       * use an `HistGradientBoostingClassifier` as a predictive model.
# - Make an hyper-parameters search using `RandomizedSearchCV` and tuning the
#   parameters:
#       * `learning_rate` with values ranging from 0.001 to 0.5. You can use
#         an exponential distribution to sample the possible values.
#       * `l2_regularization` with values ranging from 0 to 0.5. You can use
#         a uniform distribution.
#       * `max_lead_nodes` with values ranging from 5 to 30. The values should
#         be integer following a uniform distribution.
#       * `min_samples_leaf` with values ranging from 5 to 30. The values
#         should be integer following a uniform distribution.
#
# In case you have issues of with unknown categories, try to precompute the
# list of possible categories ahead of time and pass it explicitly to the
# constructor of the encoder:
#
# ```python
# categories = [data[column].unique()
#               for column in data[categorical_columns]]
# OrdinalEncoder(categories=categories)
# ```

# %% [markdown]
# ## Combining evaluation and hyper-parameters search
#
# Cross-validation was used for searching the best model parameters. We
# previously evaluate model performance through cross-validation as well. If we
# would like to combine both aspects, we need to perform a "nested"
# cross-validation. The "outer" cross-validation is applied to assess the
# model while the "inner" cross-validation set the hyper-parameters of the
# model on the data set provided by the "outer" cross-validation. In practice,
# it is equivalent of including, `GridSearchCV`, `RandomSearchCV`, or any
# `EstimatorCV` in a `cross_val_score` or `cross_validate` function call.

# %%
from sklearn.model_selection import cross_val_score

model = make_pipeline(
    preprocessor, LogisticRegressionCV(max_iter=1000, solver='lbfgs', cv=5)
)
score = cross_val_score(model, data, target, n_jobs=4, cv=5)
print(f"The accuracy score is: {score.mean():.2f} +- {score.std():.2f}")
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# Be aware that such training might involve a variation of the hyper-parameters
# of the model. When analyzing such model, you should not only look at the
# overall model performance but look at the hyper-parameters variations as
# well.
