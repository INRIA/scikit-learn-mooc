# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to scikit-learn: basic model hyper-parameters tuning
#
# The process of learning a predictive model is driven by a set of internal
# parameters and a set of training data. These internal parameters are called
# hyper-parameters and are specific for each family of models. In addition, a
# specific set of parameters are optimal for a specific dataset and thus they
# need to be optimized.
#
# This notebook shows:
# * the influence of changing model parameters;
# * how to tune these hyper-parameters;
# * how to evaluate the model performance together with hyper-parameter
#   tuning.

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv(os.path.join("..", "datasets", "adult-census.csv"))

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
    data, target, random_state=42)

# %% [markdown]
# Then, we define the preprocessing pipeline to transform differently
# the numerical and categorical data.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'native-country', 'sex']

categories = [data[column].unique()
              for column in data[categorical_columns]]

categorical_preprocessor = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer(
    [('cat-preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# %% [markdown]
# Finally, we use a tree-based classifier (i.e. histogram gradient-boosting) to
# predict whether or not a person earns more than 50,000 dollars a year.

# %%
# for the moment this line is required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    preprocessor, HistGradientBoostingClassifier(random_state=42))
model.fit(df_train, target_train)
print(f"The accuracy score using a {model.__class__.__name__} is "
      f"{model.score(df_test, target_test):.2f}")

# %% [markdown]
# ## The issue of finding the best model parameters
#
# In the previous example, we created an histogram gradient-boosting classifier
# using the default parameters by omitting to explicitely set these parameters.
#
# However, there is no reasons that this set of parameters are optimal for our
# dataset. For instance, fine-tuning the histogram gradient-boosting can be
# achieved by finding the best combination of the following parameters: (i)
# `learning_rate`, (ii) `min_samples_leaf`, and (iii) `max_leaf_nodes`.
# Nevertheless, finding this combination manually will be tedious. Indeed,
# there are relationship between these parameters which are difficult to find
# manually: increasing the depth of trees (increasing `max_samples_leaf`)
# should be associated with a lower learning-rate.
#
# Scikit-learn provides tools to explore and evaluate the parameters
# space.
# %% [markdown]
# ## Finding the best model hyper-parameters via exhaustive parameters search
#
# Our goal is to find the best combination of the parameters stated above.
#
# In short, we will set these parameters with some defined values, train our
# model on some data, and evaluate the model performance on some left out data.
# Ideally, we will select the parameters leading to the optimal performance on
# the testing set.

# %% [markdown]
# The first step is to find the name of the parameters to be set. We use the
# method `get_params()` to get this information. For instance, for a single
# model like the `HistGradientBoostingClassifier`, we can get the list such as:

# %%
print("The hyper-parameters are for a histogram GBDT model are:")
for param_name in HistGradientBoostingClassifier().get_params().keys():
    print(param_name)

# %% [markdown]
# When the model of interest is a `Pipeline`, i.e. a serie of transformers and
# a predictor, the name of the estimator will be added at the front of the
# parameter name with a double underscore ("dunder") in-between (e.g.
# `estimator__parameters`).

# %%
print("The hyper-parameters are for the full-pipeline are:")
for param_name in model.get_params().keys():
    print(param_name)

# %% [markdown]
# The parameters that we want to set are:
# - `'histgradientboostingclassifier__learning_rate'`: this parameter will
#   control the ability of a new tree to correct the error of the previous
#   sequence of trees;
# - `'histgradientboostingclassifier__max_leaf_nodes'`: this parameter will
#   control the depth of each tree.

# %% [markdown]
# ## Exercises:
#
# Use the previously defined model (called `model`) and using two nested `for`
# loops, make a search of the best combinations of the `learning_rate` and
# `max_leaf_nodes` parameters. In this regard, you will need to train and test
# the model by setting the parameters. The evaluation of the model should be
# performed using `cross_val_score`. We can propose to define the following
# parameters search:
# - `learning_rate` for the values 0.01, 0.1, and 1;
# - `max_leaf_nodes` for the values 5, 25, 45.

# %% [markdown]
# Instead of manually writting the two `for` loops, scikit-learn provides a
# class called `GridSearchCV` which implement the exhaustive search implemented
# during the exercise.
#
# Let see how to use the `GridSearchCV` estimator for doing such search.
# Since the grid-search will be costly, we will only explore the combination
# learning-rate and the maximum number of nodes.

# %%
import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {
    'histgradientboostingclassifier__learning_rate': (0.01, 0.1, 1),
    'histgradientboostingclassifier__max_leaf_nodes': (5, 43, 63),
}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=4, cv=5)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}")

# %% [markdown]
# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all hyper-parameters and their associated values. The grid-search will be in
# charge of creating all possible combinations and test them.
#
# The number of combinations will be equal to the cardesian product of the
# number of values to explore for each parameter (e.g. in our example 3 x 3
# combinations). Thus, adding new parameters with their associated values to be
# explored become rapidly computationally expensive.
#
# Once the grid-search is fitted, it can be used as any other predictor by
# calling `predict` and `predict_proba`. Internally, it will use the model with
# the best parameters found during `fit`.
#
# Get predictions for the 5 first samples using the estimator with the best
# parameters.

# %%
model_grid_search.predict(df_test.iloc[0:5])

# %% [markdown]
# You can know about these parameters by looking at the `best_params_`
# attribute.

# %%
print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")

# %% [markdown]
# In addition, we can inspect all results which are stored in the attribute
# `cv_results_` of the grid-search. We will filter some specific columns to
# from these results

# %%
# get the parameter names
column_results = [f"param_{name}"for name in param_grid.keys()]
column_results += ["mean_test_score", "std_test_score", "rank_test_score"]

cv_results = pd.DataFrame(model_grid_search.cv_results_)
cv_results = cv_results[column_results].sort_values(
    "mean_test_score", ascending=False)
cv_results = cv_results.rename(
    columns={"param_histgradientboostingclassifier__learning_rate":
             "learning-rate",
             "param_histgradientboostingclassifier__max_leaf_nodes":
             "max leaf nodes"})
cv_results

# %% [markdown]
# With only 2 parameters, we might want to visualize the grid-search as a
# heatmap. We need to transform our `cv_results` into a dataframe where the
# rows will correspond to the learning-rate values and the columns will
# correspond to the maximum number of leaf and the content of the dataframe
# will be the mean test scores.

# %%
heatmap_cv_results = cv_results.pivot_table(
    values="mean_test_score",
    index=["learning-rate"], columns=["max leaf nodes"])

import matplotlib.pyplot as plt
from seaborn import heatmap

ax = heatmap(heatmap_cv_results, annot=True, cmap="YlGnBu", vmin=0.7, vmax=0.9)
# FIXME: temporary fix since matplotlib 3.1.1 broke seaborn heatmap. Remove
# with matplotlib 3.2
ax.invert_yaxis()
_ = ax.set_ylim([0, heatmap_cv_results.shape[0]])

# %% [markdown]
# With the `GridSearchCV` estimator, the parameters need to be specified
# explicitely. We mentioned that exploring a large number of values for
# different parameters will be quickly untractable.
#
# Instead, we can randomly generate the parameter candidates. The
# `RandomSearchCV` allows for such stochastic search. It is used similarly to
# the `GridSearchCV` but the sampling distributions need to be specified
# instead of the parameter values. For instance, we will draw candidates using
# a log-uniform distribution also called reciprocal distribution. In addition,
# we will optimize 2 other parameters:
# - `max_iter`: it corresponds to the number of trees in the ensemble;
# - `min_samples_leaf`: it corresponds to the minimum number of samples
#   required in a leaf.

# %%
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


class reciprocal_int:
    def __init__(self, a, b):
        self._distribution = reciprocal(a, b)

    def rvs(self, *args, **kwargs):
        return self._distribution.rvs(*args, **kwargs).astype(int)


param_distributions = {
    'histgradientboostingclassifier__l2_regularization': reciprocal(1e-6, 1),
    'histgradientboostingclassifier__learning_rate': reciprocal(0.001, 1),
    'histgradientboostingclassifier__max_leaf_nodes': reciprocal_int(5, 63),
    'histgradientboostingclassifier__min_samples_leaf': reciprocal_int(3, 40),
}
model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    n_jobs=4, cv=5)
model_random_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_random_search.__class__.__name__} is "
    f"{model_random_search.score(df_test, target_test):.2f}")
print(
    f"The best set of parameters is: {model_random_search.best_params_}"
)

# %% [markdown]
# We can inspect the results using the attributes `cv_results` as we previously
# did.

# %%
# get the parameter names
column_results = [f"param_{name}"for name in param_distributions.keys()]
column_results += ["mean_test_score", "std_test_score", "rank_test_score"]

cv_results = pd.DataFrame(model_random_search.cv_results_)
cv_results = cv_results[column_results].sort_values(
    "mean_test_score", ascending=False)
cv_results = cv_results.rename(
    columns={"param_histgradientboostingclassifier__l2_regularization":
             "l2 regularization",
             "param_histgradientboostingclassifier__learning_rate":
             "learning-rate",
             "param_histgradientboostingclassifier__max_leaf_nodes":
             "max leaf nodes",
             "param_histgradientboostingclassifier__min_samples_leaf":
             "min samples leaf",
             "mean_test_score": "mean test accuracy",
             "rank_test_score": "ranking"})
cv_results.head()

# %% [markdown]
# In practice, a randomized grid-search is usually run with a large number of
# iterations. In order to avoid the computation cost and still make a decent
# analysis, we load the results obtained from a similar search with 200
# iterations.

# %%
import os

cv_results = pd.read_csv(
    os.path.join(
        "..", "figures", "randomized_search_results.csv"),
    index_col=0)

# %% [markdown]
# As we have more than 2 paramters in our grid-search, we cannot visualize the
# results using a heatmap. However, we can us a parallel coordinates plot.

# %%
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.drop(columns=["ranking", "std_test_score"]),
    color="mean test accuracy",
    dimensions=["learning-rate", "l2 regularization",
                "max leaf nodes", "min samples leaf",
                "mean test accuracy"],
    color_continuous_scale=px.colors.diverging.Tealrose,
)
fig.show()

# %% [markdown]
# The parallel coordinates plot will display the values of the hyper-parameters
# on different columns while the performance metric is color coded. Thus, we
# are able to quickly inspect if there is a range of hyper-parameters which is
# working or not.
#
# You can select a subset of searches by selecting for instance a range of
# value in the mean test accuracy metric.
#
# For instance, we observe that a small learning-rate (< 0.1)
# is not a good choice since a lot of the blue line (i.e. low accuracy) are
# emerging from this range of low values.

# %% [markdown]
# ## Exercises:
#
# - Build a machine learning pipeline:
#       * preprocess the categorical columns using a `OneHotEncoder` and use
#         a `StandardScaler` to normalize the numerical data.
#       * use a `LogisticRegression` as a predictive model.
# - Make an hyper-parameters search using `RandomizedSearchCV` and tuning the
#   parameters:
#       * `C` with values ranging from 0.001 to 10. You can use a reciprocal
#         distribution (i.e. `scipy.stats.reciprocal`);
#       * `solver` with possible values being `"liblinear"` and `"lbfgs"`;
#       * `penalty` with possible values being `"l2"` and `"l1"`;
#       * `drop` with possible values being `None` or `"first"`.
#
# You might get some `FitFailedWarning` and try to explain why.

# %% [markdown]
# ## Combining evaluation and hyper-parameters search
#
# Cross-validation was used for searching for the best model parameters. We
# previously evaluated model performance through cross-validation as well. If
# we would like to combine both aspects, we need to perform a "nested"
# cross-validation. The "outer" cross-validation is applied to assess the model
# while the "inner" cross-validation sets the hyper-parameters of the model on
# the data set provided by the "outer" cross-validation. In practice, it is
# equivalent to including, `GridSearchCV`, `RandomSearchCV`, or any
# `EstimatorCV` in a `cross_val_score` or `cross_validate` function call.

# %%
from sklearn.model_selection import cross_val_score

# recall the definition of our grid-search
param_distributions = {
    'histgradientboostingclassifier__max_iter': reciprocal_int(10, 50),
    'histgradientboostingclassifier__learning_rate': reciprocal(0.01, 1),
    'histgradientboostingclassifier__max_leaf_nodes': reciprocal_int(15, 35),
    'histgradientboostingclassifier__min_samples_leaf': reciprocal_int(3, 15),
}
model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    n_jobs=4, cv=5)
score = cross_val_score(model_random_search, data, target, n_jobs=4, cv=5)
print(
    f"The accuracy score is: {score.mean():.3f} +- {score.std():.3f}"
)
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# Be aware that such training might involve a variation of the hyper-parameters
# of the model. When analyzing such model, you should not only look at the
# overall model performance but look at the hyper-parameters variations as
# well.

# %% [markdown]
#
# In this notebook, we have:
# * manually tuned the hyper-parameters of a machine-learning pipeline;
# * automatically tuned the hyper-parameters of a machine-learning pipeline by
#   by exhaustively searching the best combination of parameters from a defined
#   grid;
# * automatically tuned the hyper-parameters of a machine-learning pipeline by
#   drawing values candidates from some predefined distributions;
# * integrate an hyper-parameters tuning within a cross-validation.
#
# Key ideas discussed:
# * a grid-search is a costly search and does scale with the number of
#   parameters to search;
# * a randomized-search will run with a fixed given budget;
# * when assessing the performance of a model, hyper-parameters search should
#   be computed on the training data or can be integrated within another
#   cross-validation scheme.
