# %% [markdown]
# # Automated hyperparameters tuning in scikit-learn
#
# In the previous notebook, we saw that hyperparameters can affect the
# performance of a model. In this notebook, we will show:
#
# * how to tune these hyperparameters with a grid-search and randomized search;
# * how to evaluate the model performance together with hyperparameter
#   tuning.

# %% [markdown]
# Let us reload the dataset as we did previously:

# %%
from sklearn import set_config

set_config(display="diagram")

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We extract the column containing the target

# %%
target_name = "class"
target = df[target_name]
target

# %% [markdown]
# We drop from our data the target name, and well as the fnlwgt column
# which is a censurs weighting and the education-num column, which
# duplicates the information in another column.

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
# We will define a pipeline as seen in the first module. It will handle both
# numerical and categorical features.
#
# As we will use a tree-based model as a predictor, here we apply an
# ordinal encoder on the categorical features: it encodes
# every category with an arbitrary integer. For simple models such as
# linear models, a one-hot encoder should be prefered. But for complex
# models, in particular tree-based models, the ordinal encoder is useful
# as it avoids having high-dimensional representations
#
# First we select all the categorical columns

# %%
from sklearn.compose import make_column_selector as selector


categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

# %% [markdown]
# Then we build our ordinal encoder, giving it the known categories.

# %%
from sklearn.preprocessing import OrdinalEncoder

categories = [
    data[column].unique() for column in data[categorical_columns]]

categorical_preprocessor = OrdinalEncoder(categories=categories)

# %% [markdown]
# We now use a column transformer with code to select the
# categorical columns and apply to them the ordinal encoder

# %%
from sklearn.compose import ColumnTransformer


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
     HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))])
model.fit(df_train, target_train)

# %% [markdown]
# ## Tuning using a grid-search
#
# Instead of manually writing the two `for` loops, scikit-learn provides a
# class called `GridSearchCV` which implement the exhaustive search implemented
# during the exercise.
#
# Let see how to use the `GridSearchCV` estimator for doing such search.
# Since the grid-search will be costly, we will only explore the combination
# learning-rate and the maximum number of nodes.

# %%
# %%time
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.05, 0.1, 0.5, 1, 5),
    'classifier__max_leaf_nodes': (3, 10, 30, 100)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=4, cv=2)
model_grid_search.fit(df_train, target_train)

# %% [markdown]
# Finally, we will check the accuracy of our model using the test set.

# %%
accuracy = model_grid_search.score(df_test, target_test)
print(
    f"The test accuracy score of the grid-searched pipeline is: "
    f"{accuracy:.2f}"
)

# %% [markdown]
# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all hyperparameters and their associated values. The grid-search will be in
# charge of creating all possible combinations and test them.
#
# The number of combinations will be equal to the product of the
# number of values to explore for each parameter (e.g. in our example 4 x 4
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
# The accuracy and the best parameters of the grid-searched pipeline are
# similar to the ones we found in the previous exercise, where we searched the
# best parameters "by hand" through a double for loop.
#
# In addition, we can inspect all results which are stored in the attribute
# `cv_results_` of the grid-search. We will filter some specific columns
# from these results

# %%
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
cv_results.head()

# %% [markdown]
# Let us focus on the most interesting columns and shorten the parameter
# names to remove the `"param_classifier__"` prefix for readability:

# %%
# get the parameter names
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]


# %%
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
cv_results

# %% [markdown]
# With only 2 parameters, we might want to visualize the grid-search as a
# heatmap. We need to transform our `cv_results` into a dataframe where:
#
# - the rows will correspond to the learning-rate values;
# - the columns will correspond to the maximum number of leaf;
# - the content of the dataframe will be the mean test scores.

# %%
pivoted_cv_results = cv_results.pivot_table(
    values="mean_test_score", index=["learning_rate"],
    columns=["max_leaf_nodes"])

pivoted_cv_results

# %% [markdown]
# We can use a heatmap representation to show the above dataframe visually.

# %%
import seaborn as sns
sns.set_context("talk")

ax = sns.heatmap(pivoted_cv_results, annot=True, cmap="YlGnBu", vmin=0.7,
                 vmax=0.9)
ax.invert_yaxis()

# %% [markdown]
# The above tables highlights the following things:
#
# * for too high values of `learning_rate`, the performance of the model is
#   degraded and adjusting the value of `max_leaf_nodes` cannot fix that
#   problem;
# * outside of this pathological region, we observe that the optimal choice
#   of `max_leaf_nodes` depends on the value of `learning_rate`;
# * in particular, we observe a "diagonal" of good models with an accuracy
#   close to the maximal of 0.87: when the value of `max_leaf_nodes` is
#   increased, one should increase the value of `learning_rate` accordingly
#   to preserve a good accuracy.
#
# The precise meaning of those two parameters will be explained in a latter
# notebook.
#
# For now we will note that, in general, **there is no unique optimal parameter
# setting**: 6 models out of the 16 parameter configuration reach the maximal
# accuracy (up to small random fluctuations caused by the sampling of the
# training set).

# %% [markdown]
# ## Tuning using a randomized-search
#
# With the `GridSearchCV` estimator, the parameters need to be specified
# explicitly. We already mentioned that exploring a large number of values for
# different parameters will be quickly untractable.
#
# Instead, we can randomly generate the parameter candidates. Indeed,
# such approach avoids the regularity of the grid. Hence, adding more
# evaluations can increase the resolution in each direction. This is the
# case in the frequent situation where the choice of some hyperparameters
# is not very important, as for hyperparameter 2 in the figure below.
#
# ![Randomized vs grid search](../figures/grid_vs_random_search.svg)
#
# Indeed, the number of evaluation points need to be divided across the
# two different hyperparameters. With a grid, the danger is that the
# region of good hyperparameters fall between the line of the grid: this
# region is aligned with the grid given that hyperparameter 2 has a weak
# influence. Rather, stochastic search will sample hyperparameter 1
# independently from hyperparameter 2 and find the optimal region.
#
# The `RandomizedSearchCV` class allows for such stochastic search. It is
# used similarly to the `GridSearchCV` but the sampling distributions
# need to be specified instead of the parameter values. For instance, we
# will draw candidates using a log-uniform distribution also called
# reciprocal distribution, because the parameters we are interested in
# take positive values with a natural log scaling (.1 is as close to 1 as
# 10 is).
#
# ```{note}
# Random search (with `RandomizedSearchCV`) is typical beneficial compared
# to grid search (with `GridSearchCV`) to optimize 3 or more
# hyperparameter.
# ```
#
# We will optimize 3 other parameters in addition to the ones we
# optimized above:
#
# * `max_iter`: it corresponds to the number of trees in the ensemble;
# * `min_samples_leaf`: it corresponds to the minimum number of samples
#   required in a leaf;
# * `max_bins`: it corresponds to the maximum number of bins to construct the
#   histograms.
#
# ```{note}
# The `reciprocal` function from SciPy returns a floating number. Since we
# want to us this distribution to create integer, we will create a class that
# will cast the floating number into an integer.
# ```

# %%
from scipy.stats import reciprocal


class reciprocal_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = reciprocal(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


# %% [markdown]
# Now, we can define the randomized search using the different distributions.

# %%
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'classifier__l2_regularization': reciprocal(1e-6, 1e3),
    'classifier__learning_rate': reciprocal(0.001, 10),
    'classifier__max_leaf_nodes': reciprocal_int(2, 256),
    'classifier__min_samples_leaf': reciprocal_int(1, 100),
    'classifier__max_bins': reciprocal_int(2, 255)}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    n_jobs=4, cv=5)
model_random_search.fit(df_train, target_train)

# %% [markdown]
# Then, we can compute the accuracy score on the test set.

# %%
accuracy = model_random_search.score(df_test, target_test)

print(f"The test accuracy score of the best model is " f"{accuracy:.2f}")

# %%
from pprint import pprint

print("The best parameters are:")
pprint(model_random_search.best_params_)

# %% [markdown]
# We can inspect the results using the attributes `cv_results` as we previously
# did.

# %%
# get the parameter names
column_results = [
    f"param_{name}" for name in param_distributions.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]

cv_results = pd.DataFrame(model_random_search.cv_results_)
cv_results = cv_results[column_results].sort_values(
    "mean_test_score", ascending=False)
cv_results = cv_results.rename(shorten_param, axis=1)
cv_results

# %% [markdown]
# In practice, a randomized hyperparameter search is usually run with a large
# number of iterations. In order to avoid the computation cost and still make a
# decent analysis, we load the results obtained from a similar search with 200
# iterations.

# %%
# model_random_search = RandomizedSearchCV(
#     model, param_distributions=param_distributions, n_iter=500,
#     n_jobs=4, cv=5)
# model_random_search.fit(df_train, target_train)
# cv_results =  pd.DataFrame(model_random_search.cv_results_)
# cv_results.to_csv("../figures/randomized_search_results.csv")

# %%
cv_results = pd.read_csv("../figures/randomized_search_results.csv",
                         index_col=0)

# %% [markdown]
# As we have more than 2 paramters in our grid-search, we cannot visualize the
# results using a heatmap. However, we can us a parallel coordinates plot.

# %%
(cv_results[column_results].rename(
    shorten_param, axis=1).sort_values("mean_test_score"))

# %%
import numpy as np
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.rename(shorten_param, axis=1).apply({
        "learning_rate": np.log10,
        "max_leaf_nodes": np.log2,
        "max_bins": np.log2,
        "min_samples_leaf": np.log10,
        "l2_regularization": np.log10,
        "mean_test_score": lambda x: x}),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()

# %% [markdown]
# The parallel coordinates plot will display the values of the hyperparameters
# on different columns while the performance metric is color coded. Thus, we
# are able to quickly inspect if there is a range of hyperparameters which is
# working or not.
#
# ```{note}
# We **transformed most axis values by taking a log10 or log2** to
# spread the active ranges and improve the readability of the plot.
# ```
#
# It is possible to **select a range of results by clicking and holding on
# any axis** of the parallel coordinate plot. You can then slide (move)
# the range selection and cross two selections to see the intersections.

# %% [markdown]
# ## Quizz
#
# Select the worst performing models (for instance models with a
# "mean_test_score" lower than 0.7): what do have all these models in common
# (choose one):
#
#
# |                               |      |
# |-------------------------------|------|
# | too large `l2_regularization` |      |
# | too small `l2_regularization` |      |
# | too large `learning_rate`     |      |
# | too low `learning_rate`       |      |
# | too large `max_bins`          |      |
# | too large `max_bins`          |      |
#
#
# Using the above plot, identify ranges of values for hyperparameter that
# always prevent the model to reach a test score higher than 0.86, irrespective
# of the other values:
#
#
# |                               | True | False |
# |-------------------------------|------|-------|
# | too large `l2_regularization` |      |       |
# | too small `l2_regularization` |      |       |
# | too large `learning_rate`     |      |       |
# | too low `learning_rate`       |      |       |
# | too large `max_bins`          |      |       |
# | too large `max_bins`          |      |       |

# %% [markdown]
# In this notebook, we have:
#
# * automatically tuned the hyperparameters of a machine-learning pipeline by
#   exhaustively searching the best combination from a defined grid;
# * automatically tuned the hyperparameters of a machine-learning pipeline by
#   drawing values candidates from some predefined distributions;
# * a grid-search is a costly exhaustive search and does scale with the number
#   of parameters to search;
# * a randomized-search will always run with a fixed given budget;
# * when assessing the performance of a model, hyperparameters search should
#   be tuned on the training data of a predefined train test split.
