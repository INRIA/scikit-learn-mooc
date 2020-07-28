# %% [markdown]
# # Ensemble learning: when many are better that the one
#
# In this notebook, we will go in depth into algorithms which combine several
# simple learners (e.g. decision tree, linear model, etc.) together. We will
# see that combining learners will lead to a more powerful and robust learner.
# We will focus on two families of ensemble methods:
#
# * ensemble using bootstrap (e.g. bagging and random-forest);
# * ensemble using boosting (e.g. adaptive boosting and gradient-boosting
#   decision tree).
#
# ## Benefit of ensemble method at a first glance
#
# In this section, we will give a quick demonstration on the power of combining
# several learners instead of fine-tuning a single learner.
#
# We will start by loading the "California Housing" dataset.

# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
X, y = california_housing.data, california_housing.target

# %% [markdown]
# In this dataset, we want to predict the median housing value in some district
# in California based on demographic and geographic data.

# %%
df.head()

# %% [markdown]
# We start by learning a single decision tree regressor. As we previously
# presented in the "tree in depth" notebook, this learner needs to be tuned to
# overcome over- or under-fitting. Indeed, the default parameters will not
# necessarily lead to an optimal decision tree. Instead of using the default
# value, we should search via cross-validation the value of the important
# parameters such as `max_depth`, `min_samples_split`, or `min_samples_leaf`.
#
# We recall that we need to tune these parameters because the decision trees
# tend to overfit the training data if we grow deep the trees but there are no
# rules to limit the parameters. Thus, not making a search could lead us to
# have an underfitted model.
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
# the tuning of the parameters and investigate the results.

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
# create a leaf is also equal to 1. However, the minimum number of samples to
# make a split is much higher than the default value (i.e. 50 samples).
#
# It is interesting to look at the total amount of time it took to fit all
# these different models. In addition, we can check the performance of the
# optimal decision tree on the left-out testing data.

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
# Hence, we have a model that has an $R^2$ score below 0.7. The amount of time
# to find the best learner depends on the number of fold used during the
# cross-validation in the grid-search multiplied by the cartesian product of
# the paramters combination. Therefore, the computational cost is quite high.
#
# Now we will use an ensemble method called bagging. We will later see more
# into details this method in the next section. In short, this method will use
# a base regressor (i.e. decision tree regressors) and will train several of
# them on a slightly modified version of the training set. Then, the
# predictions of all these learners will be combined by averaging.
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
# ensemble than making the parameters search of a single tree. In addition, the
# score is significantly improved with a $R^2$ close to 0.8. Furthermore, note
# that this result is obtained before any parameters tuning. This shows the
# motivation behind the use of ensemble learner: it gives a relative good
# baseline with decent performance without any parameter tuning.
#
# Now, we will go into details by presenting two ensemble families: bagging and
# boosting.
#
# ## Bagging
#
# Bagging stands for bootstrap aggregating. Indeed, it uses bootstrap samples
# to learn several models. At predict time, the predictions of each learner
# are aggregated to give the final predictions.
#
# Let's define a simple dataset that we already used in some previous notebook.

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)


def generate_data(n_samples=50, sorted=False):
    x_max, x_min = 1.4, -1.4
    len_x = x_max - x_min
    x = rng.rand(n_samples) * len_x - len_x / 2
    noise = rng.randn(n_samples) * 0.3
    y = x ** 3 - 0.5 * x ** 2 + noise
    if sorted:
        sorted_idx = np.argsort(x)
        x, y = x[sorted_idx], y[sorted_idx]
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

grid = np.linspace(np.min(x), np.max(x), num=300)
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
# A bootstrap sample corresponds to a resampling with replacement of the
# original dataset and where the size of the bootstrap sample is equal to the
# size of the original dataset. Thus, the bootstrap sample contains some
# of the data points several time and some of the original data points will not
# be present in the sample.
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
# We will generate 3 bootstrap sample and qualitatively check the difference
# with the original dataset.

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
# We observe that the 3 generated bootstrap samples are all different. To
# confirm this intuition, we can check the quantity of unique sample in the
# bootstrap samples.

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
# We can plot these decision functions on the same plot to see the difference.

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
# them, we also need to combine them. In regression, the most straightforward
# approach is to average the different predictions from all learners. We can
# plot the averaged predictions in the previous example.

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
# The plain red line shows the averaged predictions which will be the final
# preditions given by our bag of decision tree regressors.
#
# ## Random forest
#
# A popular machine-learning algorithm is random forest. Random forest is a
# modification of the bagging algorithm. In bagging, any classifier or
# regressor can be used. Random forest limits this base classifier or regressor
# to be a decision tree. In our previous example, we already used decision
# trees but we could have make the same experiment using a linear model.
#
# In addition, random forest is different from bagging when used with
# classifiers: when searching for the best split, only a subset of the original
# features are used. By default, this subset of feature is equal to the squared
# root of the original number of feature. In regression, the total number of
# available feature will be used as with the bagging regressor presented
# earlier.
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
# We see that we don't provide a `base_estimator` parameter to the random
# forest regressor. We see that our score are almost identical. Indeed, our
# problem is a regression problem and therefore, the number of features used
# in random forest and bagging is identical.
#
# When solving a classification problem, we would need to pass a tree instance
# with the parameter `max_features="sqrt"` if we want the bagging classifier
# and the random forest classifier to have the same behaviour.
#
# ### What is the difference between regressor and classifier
#
# Up to now, we only focused on regression problem. There is little difference
# between regression and classification. The first difference is that we are
# using classifiers as base estimator instead of regressors. The second
# difference lies in the way we aggregate the predictions. With the regressors,
# we were computing the averaged of the predictions. In classification, we
# will make a majority vote to give the final prediction. We can easily derived
# probabilities by looking at the prediction class counts.
#
# ## Summary
#
# We saw in this section 2 algorithms which use bootstrap samples to create
# an ensemble classifiers or regressors. The predictions are then aggregated.
# These algorithms train several classifiers or regressors on different
# bootstrap samples. This operation can be done in a very efficient manner
# since the training of each classifier or regressor can be done
# simultaneously.
#
# ## Boosting
#
# We saw that bagging builds an ensemble in a sequential manner: each learner
# is trained on an independent manner from the each other. The idea behind
# boosting is different. The ensemble is created as a sequence where the
# learner at stage `N` will require all learners from 1 to `N-1`.
#
# Intuitively, a learner will be added in the ensemble and will correct the
# mistakes done by the previous series of learners. We will start with an
# algorithm named AdaBoost to get some intuitions regarding some ideas behind
# boosting.
#
# ### Adaptive Boosting (AdaBoost)
#
# We will first present some fundamental ideas used in AdaBoost to understand
# the principle of boosting. we will use a classification problem with a
# dataset already used in the presenting the tree algorithms.

# %%
data = pd.read_csv("../datasets/penguins.csv")

# select the features of interest
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

data = data[culmen_columns + [target_column]]
data[target_column] = data[target_column].str.split().str[0]
data = data.dropna()

X, y = data[culmen_columns], data[target_column]

# %% [markdown]
# We will take from this notebook a function which plots the decision function
# learnt by the classifier.

# %%
import seaborn as sns


def plot_decision_function(X, y, clf, sample_weight=None, fit=True, ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    if fit:
        clf.fit(X, y, sample_weight=sample_weight)

    # create a grid to evaluate all possible samples
    plot_step = 0.02
    feature_0_min, feature_0_max = (
        X.iloc[:, 0].min() - 1,
        X.iloc[:, 0].max() + 1,
    )
    feature_1_min, feature_1_max = (
        X.iloc[:, 1].min() - 1,
        X.iloc[:, 1].max() + 1,
    )
    xx, yy = np.meshgrid(
        np.arange(feature_0_min, feature_0_max, plot_step),
        np.arange(feature_1_min, feature_1_max, plot_step),
    )

    # compute the associated prediction
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(
        data=pd.concat([X, y], axis=1),
        x=X.columns[0],
        y=X.columns[1],
        hue=y.name,
        ax=ax,
    )


# %% [markdown]
# As we previously did, we will train a decision tree classifier with a very
# shallow depth. We will draw the decision function obtain and we will
# show the samples for which the tree was not able to make a proper
# classification.


# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2, random_state=0)

_, ax = plt.subplots()
plot_decision_function(X, y, tree, ax=ax)

# find the misclassified samples
y_pred = tree.predict(X)
misclassified_samples_idx = np.flatnonzero(y != y_pred)

ax.plot(
    X.iloc[misclassified_samples_idx, 0],
    X.iloc[misclassified_samples_idx, 1],
    "+k",
    label="Misclassified samples",
)
ax.legend()

# %% [markdown]
# We can observe that there is several samples for which the current classifier
# was not able to make the proper decision. As we previously mentioned,
# boosting relies on creating a new classifier which will try to correct these
# error. In scikit-learn, learners usually support a parameter `sample_weight`
# which allows to specify to pay more attention to some specific samples. This
# parameters is set when calling `classifier.fit(X, y, sample_weight=weights)`.
# We will use this trick to create a new classifier by discarding all well
# classified samples and only consider the misclassified samples. Thus,
# misclassified samples will be assigned a weight of 1 while well classified
# samples will assigned to a weight of 0.

# %%
sample_weight = np.zeros_like(y, dtype=int)
sample_weight[misclassified_samples_idx] = 1

_, ax = plt.subplots()
plot_decision_function(X, y, tree, sample_weight=sample_weight, ax=ax)

ax.plot(
    X.iloc[misclassified_samples_idx, 0],
    X.iloc[misclassified_samples_idx, 1],
    "+k",
    label="Previous misclassified samples",
)
ax.legend()

# %% [markdown]
# We can see that the decision function drastically changed. Qualitatively,
# we see that the previously misclassified samples are now well classified.

# %%
y_pred = tree.predict(X)
newly_misclassified_samples_idx = np.flatnonzero(y != y_pred)
remaining_misclassified_samples_idx = np.intersect1d(
    misclassified_samples_idx, newly_misclassified_samples_idx
)

print(
    f"Number of samples previously misclassified and still misclassified: "
    f"{len(remaining_misclassified_samples_idx)}"
)

# %% [markdown]
# Now, we could think of a simple heuristic to combine together the 2
# classifiers and make some predictions using both classifiers. For instance,
# we could assign a weight to each classifier depending on the
# misclassification rate done on the dataset.

ensemble_weight = [
    (y.shape[0] - len(misclassified_samples_idx)) / y.shape[0],
    (y.shape[0] - len(newly_misclassified_samples_idx)) / y.shape[0],
]
ensemble_weight

# %% [markdown]
# So for instance, the first classifier was 94% accurate and the second one
# 69% accurate. Therefore, when predicting a class, I could trust slightly more
# the first classifier than the second one. Indeed, we could predict a weighted
# prediction using those weight.
#
# As a summary, we can note several things. First, we can learn several
# classifier which will be different by focusing more or less to some specific
# samples of the dataset. Then, boosting is different from bagging: here we
# never resample our dataset, we just assign weight to the original dataset.
#
# Finally, we see that we need some strategy to combine the learners together:
# * one need to define a way to compute the weight which need to be assigned to
#   samples;
# * one need to assign a weight to each learner when making the predictions.
#
# Here, we defined really simple scheme to define the weight. However, there is
# some statistical basis which can be used to theoretically defined these
# weights.

# %%
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
    n_estimators=3,
    algorithm="SAMME",
    random_state=0,
)
adaboost.fit(X, y)

_, axs = plt.subplots(ncols=3, figsize=(16, 6))

for ax, tree in zip(axs, adaboost.estimators_):
    plot_decision_function(X, y, tree, fit=False, ax=ax)

print(f"Weight of each classifier: {adaboost.estimator_weights_}")
print(f"Error of each classifier: {adaboost.estimator_errors_}")

# %% [markdown]
# We see that AdaBoost learnt 3 different classifiers focusing on different
# samples. Looking at the weight of each learner, we see that we are more
# confident in the first learner than the latest since the weights are
# decreasing. It is in line of the error rate of each classifier.
#
# While AdaBoost is a nice algorithm to get the internal machinery of boosting
# algorithms, this is not the most efficient machine-learning algorithm.
# The most efficient algorithm based on boosting is the gradient-boosting
# decision tree (GBDT) algorithm which we will present now.
#
# ### Gradient-boosting decision tree (GBDT)
#
# Gradient-boosting differs from AdaBoost since at each stage, it will fit
# the residuals (hence the name "gradient") instead of the samples themselves.
#
# In this section, we will provide some intuitions regarding the way learners
# will be combined to give the final prediction. In this regard, let's start
# with our regression problem.

# %%
x, y = generate_data(sorted=True)

plt.scatter(x, y, color="k", s=9)
plt.xlabel("Feature")
_ = plt.ylabel("Target")

# %% [markdown]
# As we previously discussed, boosting will be based on assembling a sequence
# of learner. We will start by creating a decision tree regressor. We will fix
# the depth of the tree such that the resulting learner will underfit the data.

# %%
tree = DecisionTreeRegressor(max_depth=3, random_state=0)
tree.fit(x.reshape(-1, 1), y)

grid = np.linspace(np.min(x), np.max(x), num=200)
y_pred_grid_raw = tree.predict(grid.reshape(-1, 1))

plt.scatter(x, y, color="k", s=9)
plt.xlabel("Feature")
plt.ylabel("Target")
line_predictions = plt.plot(grid, y_pred_grid_raw, "--")

y_pred_raw = tree.predict(x.reshape(-1, 1))
for idx in range(len(y)):
    lines_residuals = plt.plot(
        [x[idx], x[idx]], [y[idx], y_pred_raw[idx]], color="red",
    )

_ = plt.legend(
    [line_predictions[0], lines_residuals[0]], ["Fitted tree", "Residuals"]
)

# %% [markdown]
# Since the tree is underfitting the data, the accuracy of the tree will is far
# to be perfect on the training data. We can observe it on the figure by
# looking at the difference between the predictions and the ground-truth data.
# We represent these data by a red plain line that we call "Residuals".
#
# Indeed, our initial tree was not enough expressive to handle these changes.
# In a gradient-boosting algorithm, the idea will be to create a second tree
# which given the same data `x` will try to predict the residuals instead of
# `y`. Therefore, we will have a tree able to predict the error made by the
# initial tree.
#
# Let's train such a tree.

# %%
residuals = y - y_pred_raw

tree_residuals = DecisionTreeRegressor(max_depth=5, random_state=0)
tree_residuals.fit(x.reshape(-1, 1), residuals)

y_pred_grid_residuals = tree_residuals.predict(grid.reshape(-1, 1))

plt.scatter(x, residuals, color="k", s=9)
plt.xlabel("Feature")
plt.ylabel("Residuals")
line_predictions = plt.plot(grid, y_pred_grid_residuals, "--")

y_pred_residuals = tree_residuals.predict(x.reshape(-1, 1))
for idx in range(len(y)):
    lines_residuals = plt.plot(
        [x[idx], x[idx]], [residuals[idx], y_pred_residuals[idx]], color="red",
    )

_ = plt.legend(
    [line_predictions[0], lines_residuals[0]], ["Fitted tree", "Residuals"]
)

# %% [markdown]
# We see that this new tree succeed to correct the residual for some specific
# samples but not for others. We will focus on the last sample of `x` and
# explain how the predictions of both trees are combined.

# %%

_, axs = plt.subplots(ncols=2, figsize=(12, 6), sharex=True)

axs[0].scatter(x, y, color="k", s=9)
axs[0].set_xlabel("Feature")
axs[0].set_ylabel("Target")
axs[0].plot(grid, y_pred_grid_raw, "--")

axs[1].scatter(x, residuals, color="k", s=9)
axs[1].set_xlabel("Feature")
axs[1].set_ylabel("Residuals")
plt.plot(grid, y_pred_grid_residuals, "--")

for idx in range(len(y)):
    axs[0].plot(
        [x[idx], x[idx]], [y[idx], y_pred_raw[idx]], color="red",
    )
    axs[1].plot(
        [x[idx], x[idx]], [residuals[idx], y_pred_residuals[idx]], color="red",
    )

axs[0].set_xlim([1.1, 1.4])
_ = axs[1].set_xlim([1.1, 1.4])

# %% [markdown]
# For this last sample, we see that our initial tree is making an error
# (small residual). When fitting the second tree, the residual in this case is
# perfectly fitted. We can check the prediction quantitatively using the tree
# predictions. First, let's check the prediction of the initial tree and
# compare it with the true value to predict.

# %%
x_max = x[-1]
y_true = y[-1]

print(f"True value to predict for f(x={x_max:.3f}) = {y_true:.3f}")

y_pred_first_tree = tree.predict([[x_max]])[0]
print(
    f"Prediction of the first decision tree for x={x_max:.3f}: "
    f"y={y_pred_first_tree:.3f}"
)
print(f"Error of the tree: {y_true - y_pred_first_tree:.3f}")

# %% [markdown]
# As we visually observed, we have a small residual. Now, we can use the second
# tree to try to predict this residual.

# %%
print(
    f"Prediction of the residual for x={x_max:.3f}: "
    f"{tree_residuals.predict([[x_max]])[0]:.3f}"
)

# %% [markdown]
# Wee see that our second tree is capable of prediting the exact residual
# (error) that our first tree did. Therefore, we can predict the value of
# `x` by summing the prediction of the all trees in the ensemble.

# %%
y_pred_first_and_second_tree = (
    y_pred_first_tree + tree_residuals.predict([[x_max]])[0]
)
print(
    f"Prediction of the first and second decision trees combined for "
    f"x={x_max:.3f}: y={y_pred_first_and_second_tree:.3f}"
)
print(f"Error of the tree: {y_true - y_pred_first_and_second_tree:.3f}")

# %% [markdown]
# We choose a sample for which only 2 trees were enough to make the perfect
# prediction. However, we saw in the previous plot that 2 trees were not enough
# to correct the residuals for all samples. In this regard, one need to add
# several trees in the ensemble to succeed to correct the error.
#
# We will make a small comparison between random-forest and gradient boosting
# on the california housing dataset.

# %%
from sklearn.ensemble import GradientBoostingRegressor

X, y = california_housing.data, california_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,)

gradient_boosting = GradientBoostingRegressor(n_estimators=200)
start_time = time()
gradient_boosting.fit(X_train, y_train)
fit_time_gradient_boosting = time() - start_time

random_forest = RandomForestRegressor(n_estimators=200, n_jobs=-1)
start_time = time()
random_forest.fit(X_train, y_train)
fit_time_random_forest = time() - start_time

print(
    f"The performance of gradient-boosting are:"
    f"{gradient_boosting.score(X_test, y_test):.3f}"
)
print(f"Fitting time took: {fit_time_gradient_boosting:.2f} seconds")

print(
    f"The performance of random-forest are:"
    f"{random_forest.score(X_test, y_test):.3f}"
)
print(f"Fitting time took: {fit_time_random_forest:.2f} seconds")

# %% [markdown]
# In term of computation performance, the forest can be parallelized and will
# benefit from the having multiple CPU. In terms of scoring performance, the
# algorithm leads to very close results and are both really robust and
# efficient.
#
# ## Parameters consideration with random forest and gradient-boosting
#
# In the previous section, we did not focus on the parameters of random forest
# and gradient-boosting.
#
# ### Random forest
#
# The main parameters to tune with random forest is the `n_estimators`
# parameter. In general, more trees in the forest the better will be the
# performance. However, it will slow down the fitting and prediction time. So
# one have to consider limiting the number of estimators when putting such
# learner in production.
#
# The `max_depth` parameter could also be tuned. Sometimes, there is no need
# to have fully grown trees. However, be aware that with random forest, trees
# are generally deep since we are seeking to overfit each of the bootstrap
# samples which will be then mitigated by combining them. Assembling
# underfitted trees (i.e. shallow trees) might also lead to an underfitted
# forest.

# %%
param_grid = {
    "n_estimators": [10, 20, 30],
    "max_depth": [3, 5, None],
}
grid_search = GridSearchCV(
    RandomForestRegressor(n_jobs=-1), param_grid=param_grid, n_jobs=-1,
)
grid_search.fit(X_train, y_train)

columns = ["params", "mean_test_score", "rank_test_score"]
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[columns].sort_values(by="rank_test_score")

# %% [markdown]
# We can observe that in our grid-search, the larger `max_depth` with the
# largest `n_estimators` lead to the best performance.
#
# Gradient-boosting decision tree
#
# In gradient-boosting, parameters tuning is a combination of several
# parameters: `n_estimators`, `max_depth`, and `learning_rate`.
#
# Let's first discuss about the `max_depth`. We saw in the section of the
# gradient-boosting that the algorithm fit the error of the previous trees in
# the ensemble. Thus, fitting fully grown trees will be detrimental. Indeed,
# the first tree of the ensemble will overfit the data and thus no subsequent
# tree is required since there will be no residuals to be fitted. Therefore,
# the tree used in gradient-boosting will have a low depth between 3 to 8
# levels typically.
#
# Having in mind this consideration, deeper will be the trees, the residuals
# will be corrected faster and with a lower number of estimators. So
# `n_estimators` should be increased if `max_depth` is lower.
#
# Finally, we did not explain what `learning_rate` parameter was corresponding
# too. When fitting the residual one could choose if the tree should try to
# correct all possible errors or only a fraction of it. The learning-rate is
# allowing to control this behaviour. A small value of learning-rate will only
# correct the residuals of very few samples. If the learning-rate is set to 1
# then we will find a tree that fit the residuals of all samples. So, with a
# very low learning-rate, we will need more estimators to correct the error.
# However, a too large learning-rate will tend obtain an overfitted ensemble
# similarly to have a too large tree depth.

# %%
param_grid = {
    "n_estimators": [10, 30, 50],
    "max_depth": [3, 5, None],
    "learning_rate": [0.1, 1],
}
grid_search = GridSearchCV(
    GradientBoostingRegressor(), param_grid=param_grid, n_jobs=-1,
)
grid_search.fit(X_train, y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[columns].sort_values(by="rank_test_score")

# %% [markdown]
# ## Accelerate gradient-boosting
#
# We previously mentioned that random-forest is an efficient algorithm since
# each tree of the ensemble can be fitted at the same time from an independent
# manner.
#
# In gradient-boosting, the algorithm is a sequential algorithm. it requires
# the `N-1` trees to fit the tree at the stage `N`. Therefore, the algorithm
# is quite computationally expensive. The most expensive in this algorithm is
# indeed the search for the best split which is a brute-force approach: all
# possible split are evaluated and the best one is picked. We explain this
# process in the notebook presenting the tree algorithm to which you can refer.
#
# To accelerate the gradient-boosting algorithm, one could reduce the number
# of split to be evaluated. Thus, the score of such tree will be much lower.
# However, since we are combining several trees in a gradient-boosting, we are
# just required to add enough estimator to overcome this issue.
#
# This algorithm is called `HistGradientBoostingClassifier` and
# `HistGradientBoostingRegressor`. Indeed, the dataset `X` is first binned
# by computing histograms to make the split. The number of splits to evaluate
# is much smaller. This algorithm becomes extremely efficient when the dataset
# has 10,000+ samples.
#
# We are giving an example of such dataset and we can compare it with the
# earlier experiment in the previous section.

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

histogram_gradient_boosting = HistGradientBoostingRegressor(
    max_iter=200, random_state=0,
)
start_time = time()
histogram_gradient_boosting.fit(X_train, y_train)
fit_time_histogram_gradient_boosting = time() - start_time

print(
    f"The performance of histogram gradient-boosting are:"
    f"{histogram_gradient_boosting.score(X_test, y_test):.3f}"
)
print(f"Fitting time took: {fit_time_histogram_gradient_boosting:.2f} seconds")

# %% [markdown]
# The histogram gradient-boosting is the best algorithm in term of score.
# It will also scale whenever the number of samples increase while the normal
# gradient-boosting will not.
#
# ## Wrap-up
#
# So in this notebook we presented ensemble learners which are a type of
# learners which combined simpler learner together. We saw 2 strategies:
# one base on bootstrap samples allowing to fit learner in a parallel manner
# and the other called boosting which fit learners in a sequential manner.
#
# From these two families, we mainly focus on giving intuitions regarding the
# internal machinery of the random forest and gradient-boosting algorithms
# which are state-of-the-art methods.
