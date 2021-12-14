# %% [markdown]
# # Hyperparameter tuning by randomized-search
#
# In the previous notebook, we showed how to use a grid-search approach to
# search for the best hyperparameters maximizing the generalization performance
# of a predictive model.
#
# However, a grid-search approach has limitations. It does not scale when
# the number of parameters to tune is increasing. Also, the grid will impose
# a regularity during the search which might be problematic.
#
# In this notebook, we will present another method to tune hyperparameters
# called randomized search.

# %% [markdown]
# ## Our predictive model
#
# Let us reload the dataset as we did previously:

# %%
from sklearn import set_config

set_config(display="diagram")

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We extract the column containing the target.

# %%
target_name = "class"
target = adult_census[target_name]
target

# %% [markdown]
# We drop from our data the target and the `"education-num"` column which
# duplicates the information with `"education"` columns.

# %%
data = adult_census.drop(columns=[target_name, "education-num"])
data.head()

# %% [markdown]
# Once the dataset is loaded, we split it into a training and testing sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# We will create the same predictive pipeline as seen in the grid-search
# section.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4)),
])

model

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
# will draw candidates using a log-uniform distribution because the parameters
# we are interested in take positive values with a natural log scaling (.1 is
# as close to 1 as 10 is).
#
# ```{note}
# Random search (with `RandomizedSearchCV`) is typically beneficial compared
# to grid search (with `GridSearchCV`) to optimize 3 or more
# hyperparameters.
# ```
#
# We will optimize 3 other parameters in addition to the ones we
# optimized in the notebook presenting the `GridSearchCV`:
#
# * `l2_regularization`: it corresponds to the constant to regularized the loss
#   function
# * `min_samples_leaf`: it corresponds to the minimum number of samples
#   required in a leaf;
# * `max_bins`: it corresponds to the maximum number of bins to construct the
#   histograms.
#
# We recall the meaning of the 2 remaining parameters:
#
# * `learning_rate`: it corresponds to the speed at which the gradient-boosting
#   will correct the residuals at each boosting iteration;
# * `max_leaf_nodes`: it corresponds to the maximum number of leaves for each
#   tree in the ensemble.
#
# ```{note}
# `scipy.stats.loguniform` can be used to generate floating numbers. To
# generate random values for integer-valued parameters (e.g.
# `min_samples_leaf`) we can adapt is as follows:
# ```

# %%
from scipy.stats import loguniform


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


# %% [markdown]
#
# Now, we can define the randomized search using the different distributions.
# Executing 10 iterations of 5-fold cross-validation for random
# parametrizations of this model on this dataset can take from 10 seconds to
# several minutes, depending on the speed of the host computer and the number
# of available processors.

# %%
%%time
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'classifier__l2_regularization': loguniform(1e-6, 1e3),
    'classifier__learning_rate': loguniform(0.001, 10),
    'classifier__max_leaf_nodes': loguniform_int(2, 256),
    'classifier__min_samples_leaf': loguniform_int(1, 100),
    'classifier__max_bins': loguniform_int(2, 255),
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    cv=5, verbose=1,
)
model_random_search.fit(data_train, target_train)

# %% [markdown]
# Then, we can compute the accuracy score on the test set.

# %%
accuracy = model_random_search.score(data_test, target_test)

print(f"The test accuracy score of the best model is "
      f"{accuracy:.2f}")

# %%
from pprint import pprint

print("The best parameters are:")
pprint(model_random_search.best_params_)

# %% [markdown]
#
# We can inspect the results using the attributes `cv_results` as we did
# previously.


# %%
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


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
#     model, param_distributions=param_distributions, n_iter=200,
#     n_jobs=2, cv=5)
# model_random_search.fit(data_train, target_train)
# cv_results =  pd.DataFrame(model_random_search.cv_results_)
# cv_results.to_csv("../figures/randomized_search_results.csv")

# %%
cv_results = pd.read_csv("../figures/randomized_search_results.csv",
                         index_col=0)

# %% [markdown]
# As we have more than 2 parameters in our grid-search, we cannot visualize the
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
#
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
# In particular for this hyper-parameter search, it is interesting to see that
# the yellow lines (top performing models) all reach intermediate values for
# the learning rate, that is, tick values between -2 and 0 which correspond to
# learning rate values of 0.01 to 1.0 once we invert the log10 transform for
# that axis.
#
# It is possible to **select a range of results by clicking and holding on any
# axis** of the parallel coordinate plot. You can then slide (move) the range
# selection and cross two selections to see the intersections. You can undo a
# selection by clicking once again on the same axis.
#
# We also observe that it is not possible to select the highest performing
# models by selecting lines of on the `max_bins` axis with tick values between
# 1 and 3.
#
# The other hyper-parameters are not very sensitive. We can check that if we
# select the `learning_rate` axis tick values between -1.5 and -0.5 and
# `max_bins` tick values between 5 and 8, we always select top performing
# models, whatever the values of the other hyper-parameters.

# %% [markdown]
#
# In this notebook, we have seen how randomized search offer a valuable
# alternative to grid-search when the number of hyperparameters to tune is more
# than two. It also alleviates the regularity imposed by the grid that might be
# problematic sometimes.
