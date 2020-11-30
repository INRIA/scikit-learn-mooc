# %% [markdown]
# # 📃 Solution for Exercise 05
#
# The aim of the exercise is to get familiar with the histogram
# gradient-boosting in scikit-learn. Besides, we will use this model within
# a cross-validation framework in order to inspect internal parameters found
# via grid-search.
#
# We will use the california housing dataset.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# %% [markdown]
# First, create a histogram gradient boosting regressor. You can set the number
# of trees to be large enough. Indeed, you fix the parameter such that model
# will use early-stopping.

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

hist_gbdt = HistGradientBoostingRegressor(
    max_iter=1000, early_stopping=True, random_state=0)

# %% [markdown]
# We will use a grid-search to find some optimal parameter for this model.
# In this grid-search, you should search for the following parameters:
#
# * `max_depth: [3, 8]`;
# * `max_leaf_nodes: [15, 31]`;
# * `learning_rate: [0.1, 1]`.
#
# Feel free to explore more the space with additional values. Create the
# grid-search providing the previous gradient boosting instance as model.

# %%
from sklearn.model_selection import GridSearchCV

params = {
    "max_depth": [3, 8],
    "max_leaf_nodes": [15, 31],
    "learning_rate": [0.1, 1],
}

search = GridSearchCV(hist_gbdt, params)

# %% [markdown]
# Finally, we will run our experiment through cross-validation. In this regard,
# define a 5-fold cross-validation. Besides, be sure to shuffle the the data.
# Subsequently, use the function `sklearn.model_selection.cross_validate`
# to run the cross-validation. You should as well set `return_estimator=True`,
# such that we can investigate the inner model trained via cross-validation.

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
results = cross_validate(
    search, X, y, cv=cv, return_estimator=True, n_jobs=-1)

# %% [markdown]
# We got the results of the cross-validation. First check what is the mean and
# standard deviation score.

# %%
print(f"R2 score with cross-validation:\n"
      f"{results['test_score'].mean():.3f} +/- "
      f"{results['test_score'].std():.3f}")

# %% [markdown]
# Then inspect the `estimator` entry of the results and check the best
# parameters values. Besides, check the number of trees used by the model.

# %%
for estimator in results["estimator"]:
    print(estimator.best_params_)
    print(f"# trees: {estimator.best_estimator_.n_iter_}")

# %% [markdown]
# We observe that the parameters are varying. We can get the intuition that
# results of the inner CV are very close for certain set of parameters.

# %% [markdown]
# Inspect the results of the inner CV for each estimator of the outer CV.
# Aggregate the mean test score for each parameter combination and make a box
# plot of these scores.

# %%
import pandas as pd

index_columns = [f"param_{name}" for name in params.keys()]
columns = index_columns + ["mean_test_score"]

inner_cv_results = []
for cv_idx, estimator in enumerate(results["estimator"]):
    search_cv_results = pd.DataFrame(estimator.cv_results_)
    search_cv_results = search_cv_results[columns].set_index(index_columns)
    search_cv_results = search_cv_results.rename(
        columns={"mean_test_score": f"CV {cv_idx}"})
    inner_cv_results.append(search_cv_results)
inner_cv_results = pd.concat(inner_cv_results, axis=1).T
inner_cv_results.columns = inner_cv_results.columns.to_flat_index()

# %%
import seaborn as sns
sns.set_context("talk")

ax = sns.boxplot(
    data=inner_cv_results, orient="h", color="tab:blue", whis=100)
ax.set_xlabel("R2 score")
ax.set_ylabel("Parameters")
_ = ax.set_title("Inner CV results with parameters\n"
                 "(max_depth, max_leaf_nodes, learning_rate)")

# %% [markdown]
# We see that the first 4 first ranked set of parameters are very close.
# Indeed, one would select any of these 4 combinations just due to random
# variations. It coincides with the results that we observe when inspecting the
# best parameters of the outer CV.
