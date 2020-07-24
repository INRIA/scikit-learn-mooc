# %% [markdown]
# # Ensemble learning: when many are better that the one
#
# In this notebook, we will go in depth into algorithms which combine several
# simple models (e.g. decision tree, linear model, etc.) together. We will see
# that combining learners will lead to more powerful learner which are more
# robust. We will see into details 2 family of ensemble:
#
# * ensemble using bootstrap (e.g. bagging and random-forest);
# * ensemble using boosting (e.g. gradient-boosting decision tree).
#
# ## Benefit of ensemble method at a first glance
#
# In this section, we will give a quick demonstration on the power of combining
# several learners instead of fine-tuning a single learner.
#
# We will start by loading data from "California Housing".

# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
X, y = california_housing.data, california_housing.target

# %% [markdown]
# In this dataset, we try to predict the median housing value in some district
# in California based on demographic and geographic data

# %%
df.head()

# %% [markdown]
# We will learn a single decision tree regressor. As we previously presented
# in the notebook presenting the decision tree, this learner need to be tuned.
# Indeed, the default parameter will not lead to an optimal decision tree.
# Instead of using the default value, we should search via cross-validation
# the value of the important parameter such as `max_depth`,
# `min_samples_split`, or `min_samples_leaf`. We recall that we need to tune
# these parameters because the decision trees tend to overfit the training data
# if we grow deep the trees but there are no rules to limit the parameters.
# Thus, not making a search could lead us to have underfitted model.
#
# First, let's keep a set of data to test our final model.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0,
)

# %% [markdown]
# We will first make a grid-search to fine-tune the parameters that we
# mentioned earlier.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 10, 30, 50],
    "min_samples_leaf": [0.01, 0.05, 0.1, 1],
}
cv = 3
tree = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
)

tree.fit(X_train, y_train)

# %% [markdown]
# We can create a dataframe storing the important information collected during
# the parameter tuning and investigate the results.

# %%
import pandas as pd

cv_results = pd.DataFrame(tree.cv_results_)
interesting_columns = [
    "param_max_depth",
    "param_min_samples_split",
    "param_min_samples_leaf",
    "mean_test_score",
    "rank_test_score",
    "mean_fit_time",
]
cv_results = cv_results[interesting_columns].sort_values(by="rank_test_score")
cv_results

# %% [markdown]
# From theses results, we can see that the best parameters is a combination
# where the depth of the tree is not limited, the minimum number of samples to
# make a leaf is also equal to 1. However, the minimum number of samples to
# make a split is much higher than the default value (i.e. 50 samples).
#
# To run this grid-search, we can check the total amount of time it took to
# fit all these different model. In addition, we can check the performance
# of the optimal decision tree on the left-out testing data.

# %%
total_fitting_time = (cv_results["mean_fit_time"] * cv).sum()
print(
    f"Required training time of the GridSearchCV: "
    f"{total_fitting_time:.2f} seconds"
)
print(
    f"Best R2 score of a single tree: {tree.best_score_:.3f}"
)

# %% [markdown]
# Hence, we have a model that has an $R^2$ score below 0.7. In addition,
# the amount of time to find this learner depends on the number of fold used
# during the cross-validation in the grid-search multiplied by the cartesian
# product of the paramters combination. Therefore, the computational cost is
# quite high.
#
# Now we will use an ensemble method called bagging. We will see more into
# details this method in the next section. In short, this method will use
# a base regressor (i.e. decision tree regressors) and will train several of
# them on a slightly modified version of the training set. Then, the
# predictions given by each tree will be combine by averaging.
#
# Here, we will use 50 decision trees and check the fitting time as well as
# the performance on the left-out testing data. It is important to note that
# we are not going to tune any parameter of the decision tree.

# %%
from time import time
from sklearn.ensemble import BaggingRegressor

base_estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
    base_estimator=base_estimator, n_estimators=50, random_state=0,
)

start_fitting_time = time()
bagging_regressor.fit(X_train, y_train)
elapsed_fitting_time = time() - start_fitting_time

print(f"Elapsed fitting time: {elapsed_fitting_time:.2f} seconds")
print(f"R2 score: {bagging_regressor.score(X_test, y_test):.3f}")

# %% [markdown]
# We can see that the training time is indeed much shorter to train the full
# ensemble than making the parameter search of a single tree. In addition,
# the score is largely improved with a $R^2$ close to 0.8 while we did not
# tune any of the parameters. This shows the motivation behind the use of
# ensemble learner.
#
# Now, we will go into details by presenting 2 ensemble families: bagging and
# boosting.
#
# ## Bagging
#
# We will first focus on the bagging algorithm. Bagging stands for bootstrap
# aggregating. Indeed, it uses bootstrap samples to learn several learners.
# At predict time, the predictions of each learner are aggregated to give the
# final predictions.
#
# Let's define a simple dataset that we already used in some previous notebook.

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)


def generate_data(n_samples=50):
    x_max, x_min = 1.4, -1.4
    len_x = x_max - x_min
    x = rng.rand(n_samples) * len_x - len_x / 2
    noise = rng.randn(n_samples) * 0.3
    y = x ** 3 - 0.5 * x ** 2 + noise
    return x, y


x, y = generate_data(n_samples=50)

plt.scatter(x, y,  color='k', s=9)
plt.xlabel("Feature")
_ = plt.ylabel("Target")

# %% [markdown]
# The link between our feature and the target to predict is non-linear.
# However, a decision tree is capable to fit such data

# %%
tree = DecisionTreeRegressor(max_depth=3, random_state=0)
tree.fit(x.reshape(-1, 1), y)

grid = np.linspace(np.min(x), np.max(x), num=30)
y_pred = tree.predict(grid.reshape(-1, 1))

plt.scatter(x, y, color="k", s=9)
plt.plot(grid, y_pred, label="Tree fitting")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()

# %% [markdown]
# Let's see how we can use bootstraping to learn several trees.
#
# ### Bootstrap sample
#
# Let's first define what is a bootstrap sample: it is a resampling with
# replacement of the same size than the original sample. Thus, the bootstrap
# sample will contain some of the data points several time and some of the
# original data points will not be present in the sample.
#
# We will create a function that given `x` and `y` will return a bootstrap
# sample `x_bootstrap` and `y_bootstrap`.


# %%
def bootstrap_sample(x, y):
    bootstrap_indices = rng.choice(
        np.arange(y.shape[0]), size=y.shape[0], replace=True,
    )
    x_bootstrap_sample = x[bootstrap_indices]
    y_bootstrap_sample = y[bootstrap_indices]
    return x_bootstrap_sample, y_bootstrap_sample


# %% [markdown]
# Let's generate 3 bootstrap sample and check the difference with the original
# dataset.


# %%
n_bootstrap = 3
_, axs = plt.subplots(
    ncols=n_bootstrap, figsize=(16, 6), sharex=True, sharey=True
)

for idx, (ax, _) in enumerate(zip(axs, range(n_bootstrap))):
    x_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(x, y)
    ax.scatter(
        x_bootstrap_sample, y_bootstrap_sample,
    )
    ax.set_title(f"Bootstrap sample #{idx}")
    ax.set_ylabel("Target")
    ax.set_xlabel("Feature")

# %% [markdown]
# We can observe that the 3 generated bootstrap are all different. We can even
# check the amount of samples in the bootstrap sample which are present in the
# original dataset.

# %%
# we need to generate a larger set to have a good estimate
x_huge, y_huge = generate_data(n_samples=10000)
x_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(x_huge, y_huge)

print(
    f"Percentage of samples present in the original dataset: "
    f"{np.unique(x_bootstrap_sample).size / x_bootstrap_sample.size * 100:.1f}"
    f"%"
)

# %% [markdown]
# Theoretically, 63.2% of the original data points of the original dataset will
# be present in the bootstrap sample. The other 36.8% are just some repeated
# samples.
#
# So, we are able to generate as many datasets, all slightly different. Now,
# we can fit a decision tree for each of these datasets and they will be all
# slightly different as well.

# %%
_, axs = plt.subplots(
    ncols=n_bootstrap, figsize=(16, 6), sharex=True, sharey=True,
)

forest = []
for idx, (ax, _) in enumerate(zip(axs, range(n_bootstrap))):
    x_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(x, y)
    ax.scatter(x_bootstrap_sample, y_bootstrap_sample)

    forest.append(
        DecisionTreeRegressor(max_depth=3, random_state=0).fit(
            x_bootstrap_sample.reshape(-1, 1), y_bootstrap_sample
        )
    )

    grid = np.linspace(np.min(x), np.max(x), num=300)
    y_pred = forest[-1].predict(grid.reshape(-1, 1))
    ax.plot(grid, y_pred, linewidth=3, label="Fitted tree")

    ax.legend()
    ax.set_ylabel("Target")
    ax.set_xlabel("Features")
    ax.set_title(f"Bootstrap #{idx}")

# %% [markdown]
# We can plot all trees decision function on the same plot to highlight that
# they are indeed different.

# %%
_, ax = plt.subplots()
ax.scatter(x, y, color="k", alpha=0.4)
y_pred_forest = []
for tree_idx, tree in enumerate(forest):
    y_pred = tree.predict(grid.reshape(-1, 1))
    ax.plot(
        grid,
        y_pred,
        "--",
        label=f"Tree #{tree_idx} predictions",
        linewidth=3,
        alpha=0.8,
    )
    y_pred_forest.append(y_pred)

plt.xlabel("Feature")
plt.ylabel("Target")
_ = plt.legend()

# %% [markdown]
# ### Aggregating
#
# Once that our trees are fitted and we are able to get predictions for each of
# them, we also need to combine these predictions. The most straightforward
# approach is to average the different predictions. We can plot the average
# predictions on the plot that we shown previously.

# %%
_, ax = plt.subplots()
ax.scatter(x, y, color="k", alpha=0.4)
y_pred_forest = []
for tree_idx, tree in enumerate(forest):
    y_pred = tree.predict(grid.reshape(-1, 1))
    ax.plot(
        grid,
        y_pred,
        "--",
        label=f"Tree #{tree_idx} predictions",
        linewidth=3,
        alpha=0.5,
    )
    y_pred_forest.append(y_pred)

# Averaging the predictions
y_pred_forest = np.mean(y_pred_forest, axis=0)
ax.plot(
    grid,
    y_pred_forest,
    "-",
    label="Averaged predictions",
    linewidth=3,
    alpha=0.8,
)

plt.xlabel("Feature")
plt.ylabel("Target")
_ = plt.legend()

# %% [markdown]
# The red line on the plot is showing what would be the final predictions of
# our model.
#
# ## Random Forest
#
# One of the widely used algorithm in machine learning is Random Forest.
# Random Forest is a modification of the bagging algorithm. In bagging, any
# classifier or regressor can be used. Random forest limit this base classifier
# or regressor to be a decision tree, similarly to what we shown in the
# previous section. There is a noticeable difference in classification: when
# searching for the best split, only a subset of the original feature will be
# used. This subset will be equal to the squared root of the original number
# of feature. In regression, the total number of available feature will be
# used.
#
# We will illustrate the usage of a random forest and compare it with the
# bagging regressor on the "California housing" dataset.

# %%
X, y = california_housing.data, california_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(
    n_estimators=100, random_state=0, n_jobs=-1
)
bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(random_state=0),
    n_estimators=100,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)
bagging.fit(X_train, y_train)

print(
    f"Performance of random forest: {random_forest.score(X_test, y_test):.3f}"
)
print(f"Performance of bagging: {bagging.score(X_test, y_test):.3f}")

# %% [markdown]
# First, we can observe that we cannot give the `base_estimator` to the
# random forest regressor since it is already fixed. Finally, we can see that
# the score on the testing set is exactly the same. In classification setting,
# we would need to pass a tree instance with the parameter
# `max_features="sqrt"` if we want the bagging classifier and random forest
# classifier to behave the same.
