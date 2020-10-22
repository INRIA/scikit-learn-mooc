# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Evaluation of your predictive model: the adequate framework.

# %% [markdown]
# ## Introduction
# In the previous notebooks, we check how to fit a machine-learning model. When
# we evaluate the performance of our model, we did not go into details into
# the evaluation framework that one should use in machine-learning.
#
# In this notebook, we will present the cross-validation framework and
# emphasize the importance of evaluating a model in such framework.
# In addition, we will show a couple of example of good practices to always
# apply such as nested cross-validation when the parameter of a model should be
# fine-tuned.

# %% [markdown]
# ## Train and test datasets
# Before to go in the cross-validation framework, we are going to linger on the
# necessity to always have a training and testing sets. Let's first look at
# the limitation of using a unique dataset.
#
# ### Load the California housing dataset

# %%
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

# %% [markdown]
# We use this dataset to predict the median value of house in an area in
# California. The feature collected are based on general real-estate
# and geographical information.

# %%
print(housing.DESCR)

# %%
X.head()

# %% [markdown]
# To simplify future visualization, we transform the target such that it is
# later shown in k$

# %%
y *= 100
y.head()

# %% [markdown]
# ### Emperical error vs. generalization error
# As mentioned previously, we start by fitting a decision tree regressor on the
# full dataset.

# %%
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# %% [markdown]
# Once our regressor is trained, we would like to know the potential
# performance of our regressor once we will deploy it in production. For this
# purpose, we use the mean absolute error which give us an error in the native
# unit of the target, i.e. k$.

#  %%
from sklearn.metrics import mean_absolute_error

y_pred = regressor.predict(X)
score = mean_absolute_error(y_pred, y)
print(f"In average, our regressor make an error of {score:.2f} k$")

# %% [markdown]
# Such results are too optimistic to be true. Indeed, we trained and predict
# on the same dataset. Since our decision tree was fully grown, every samples
# in the dataset is potentially a node. Therefore, our decision tree memorize
# the dataset given during `fit`.
#
# This error computed is called the **emperical error**. In some sort, we try
# to learn a predictive model which minimize this error but that at the same
# time should minimize an error on an unseen dataset. This error is also called
# the **generalization error**. Thus, the basic evaluation involves:
#
# * spliting our dataset into two subsets: a training set and a testing set;
# * estimating the emperical error on the training set and the generalization
#   error on the testing set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# %%
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_train)
score = mean_absolute_error(y_pred, y_train)
print(f"The emperical error of our model is {score:.2f} k$")

# %%
y_pred = regressor.predict(X_test)
score = mean_absolute_error(y_pred, y_test)
print(f"The generalization error of our model is {score:.2f} k$")

# %% [markdown]
# This setup emulate the training-production setup. We used a traning set to
# learn a model and predict on unseen data. But we can compute the error on
# the unseen data predictions because, we kept aside the true labels.
#
# However, the previous framework does not give any indication regarding the
# robustness of our predictive model. We could have been lucky while splitting
# our dataset and the generalization error could be over-optimistic.
#
# Cross-validation allows to estimate the robustness of a predictive model
# by repeating the splitting procedure. It will give several
# emperical-generalization errors and thus some confidence interval of the
# model performance. Note that we could defined different splitting procedures.
# This is indeed the different types of cross-validation that we will see
# later.
#
# The simplest strategy is to shuffle our data and split into two sets has we
# previously did and repeat several times fit/predict. In scikit-learn, using
# the function `cross_validate` with the cross-validation `ShuffleSplit` allows
# us to make such evaluation.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)

result_cv = cross_validate(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
)
result_cv = pd.DataFrame(result_cv)
# revert the negation to get the error and not the negative score
result_cv["test_score"] *= -1

# %% [markdown]
# Let's check the results reported by the cross-validation.

# %%
result_cv.head()

# %% [markdown]
# We get timing information to fit and predict at each round of
# cross-validation. In addition, we get the test score which corresponds to the
# generalization error on each of the split.

# %%
len(result_cv)

# %% [markdown]
# We get 30 entries in our resulting dataframe because we performed 30 splits.
# Therefore, we can show the distribution of the generalization error and
# thus have an estimate of its variance.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(result_cv["test_score"], kde=True, bins=20)
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# We observe that the generalization error is centered around 45.5 k$ and range
# from 44 k$ to 47 k$. While this information is interesting, it is not enough
# to conclude either that our evaluation or our model are working.
#
# To know if we should trust our evaluation, we should access the variance of
# the generalization error. If the variance is large, it means that we cannot
# conclude anything about the model performance. If the variance is small then
# we are confident on the reported error and we can safely interpret them.
#
# To assess the variance of the generalization error, we plot the target
# distribution.

# %%
sns.displot(y, kde=True, bins=20)
plt.xlabel("Median House Value (k$)")
print(f"The target variance is: {y.var():.2f} k$")

# %% [markdown]
# We observe that the target ranges from 0 k$ up to 500 k$ and we also reported
# the variance. Now, we can check the same statistic with the generalization
# error.

# %%
print(
    f"The variance of the generalization error is: "
    f"{result_cv['test_score'].var():.2f} k$"
)
# %% [markdown]
# We observe that the variance of the generalization error is indeed small
# in comparison with the target variance. So we can safely trust the reported
# results.
#
# Now, let's check if our predictive model give us good performance. We recall
# that our model makes in average an error of 45 k$. With this piece of
# information and looking at the target distribution, such error might be fine
# when predicting houses with a value of 500 k$. However, it would be an issue
# with house with a value a 50 k$. Thus, this indicate that our metric is not
# ideal. We should have take a metric which will be relative to the target
# value to predict: the mean absolute percentage error would have been a much
# better choice.
#
# But in all case, an error of 45 k$ might be to large to use our model to
# automatically tag house value without expert supervision.
#
# We should note that it is also interesting to compare the generalization
# error with the emperical error. Thus, we need to compute the error on the
# training set which is possible using the `cross_validate` function.

# %%
result_cv = cross_validate(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
    return_train_score=True,
)
result_cv = pd.DataFrame(result_cv)

# %%
scores = result_cv[["train_score", "test_score"]] * -1
sns.histplot(scores, bins=50)
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# So by plotting the distribution of the emperical and generalization errors,
# it gives us information about under- or over-fitting of our predictive model.
# Here, having an small emperical error and a large generalization error is
# typical from a predictive model that overfit.
#
# The hyper-parameter of a model is usually the key to go from a model that
# underfit to a model that overfit. We can acquire knowledge by plotting a
# curve called the validation curve. This curve apply the above experiment
# and vary the value of an hyper-parameter.
#
# For the decision tree, the `max_depth` parameter controls the trade-off
# between under-/over-fitting.

# %%
from sklearn.model_selection import validation_curve

max_depth = [1, 5, 10, 15, 20]
train_scores, test_scores = validation_curve(
    regressor, X, y,
    param_name="max_depth", param_range=max_depth,
    cv=cv, scoring="neg_mean_absolute_error",
    n_jobs=-1,
)

# %%
_, ax = plt.subplots()
ax.plot(
    max_depth, -train_scores.mean(axis=1),
    linestyle="-.", label="Emperical error",
    alpha=0.8,
)
ax.fill_between(
    max_depth,
    -train_scores.mean(axis=1) - train_scores.std(axis=1),
    -train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.5,
    label="Var. emperical error"
)
ax.plot(
    max_depth, -test_scores.mean(axis=1),
    linestyle="-.", label="Generalization error",
    alpha=0.8,
)
ax.fill_between(
    max_depth,
    -test_scores.mean(axis=1) - test_scores.std(axis=1),
    -test_scores.mean(axis=1) + test_scores.std(axis=1),
    alpha=0.5,
    label="Var. generalization error"
)

ax.set_xticks(max_depth)
ax.set_xlabel("Maximum depth of decision tree")
ax.set_ylabel("Mean absolute error (k$)")
ax.set_title("Validation curve for decision tree")
_ = plt.legend()

# %% [markdown]
# The validation curve can be divided into 3 areas. For `max_depth < 10`,
# the decision tree clearly underfit. Both emperical and generalization errors
# are high. For `max_depth=10` corresponds to the parameter for which the
# decision tree generalizes the best. For `max_depth > 10`, the decision tree
# overfit. The emperical error becomes small while the generalization error
# increases.
#
# In the analysis that we carried out above, we were lucky because the variance
# of the errors were small. Now, we will focus on one factor that can affect
# this variance: the size of the dataset.
#
# ### Effect of the sample size on the variance analysis
# We are quite lucky. Our dataset count many samples.

# %%
y.size

# %% [markdown]
# Let's make an experiment and reduce the number of samples and repeat the
# previous experiment.


# %%
def make_cv_analysis(regressor, X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    result_cv = pd.DataFrame(
        cross_validate(
            regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
            return_train_score=True
        )
    )
    return (result_cv["test_score"] * -1).values

# %%
import numpy as np

sample_sizes = [100, 500, 1000, 5000, 10000, 15000, y.size]

scores_sample_sizes = {"# samples": [], "test score": []}
rng = np.random.RandomState(0)
for n_samples in sample_sizes:
    sample_idx = rng.choice(np.arange(y.size), size=n_samples, replace=False)
    X_sampled, y_sampled = X.iloc[sample_idx], y[sample_idx]
    score = make_cv_analysis(regressor, X_sampled, y_sampled)
    scores_sample_sizes["# samples"].append(n_samples)
    scores_sample_sizes["test score"].append(score)

scores_sample_sizes = pd.DataFrame(
    np.array(scores_sample_sizes["test score"]).T,
    columns=scores_sample_sizes["# samples"]
)

# %%
sns.displot(scores_sample_sizes, kind="kde")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title(
    "Generalization errors distribution \nby varying the sample size"
)

# %% [markdown]
# For the different sample size, we plotted the distribution of the
# generalization error. We observe that smaller is the sample size, larger is
# the variance of the generalization errors. Thus, having a small number of
# samples might put us in the situation where it is impossible to get a
# reliable evaluation.
#
# Here, we plotted the different curve to highlight the issue of small sample
# size. However, this experiment is also used to draw the so-called
# "learning curve". This curve give some additional indication regarding the
# benefit of adding new training samples to improve the performance of a model.

# %%
from sklearn.model_selection import learning_curve

results = learning_curve(
    regressor, X, y, train_sizes=sample_sizes[:-1],
    cv=cv, scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
train_size, train_scores, test_scores = results[:3]

# %%
_, ax = plt.subplots()
ax.plot(
    train_size, train_scores.mean(axis=1),
    linestyle="-.", label="Emperical error",
    alpha=0.8,
)
ax.fill_between(
    train_size,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.5,
    label="Var. emperical error"
)
ax.plot(
    train_size, test_scores.mean(axis=1),
    linestyle="-.", label="Generalization error",
    alpha=0.8,
)
ax.fill_between(
    train_size,
    test_scores.mean(axis=1) - test_scores.std(axis=1),
    test_scores.mean(axis=1) + test_scores.std(axis=1),
    alpha=0.5,
    label="Var. generalization error"
)

ax.set_xticks(train_size)
ax.set_xscale("log")
ax.set_xlabel("Number of samples in the training set")
ax.set_ylabel("Mean absolute error (k$)")
ax.set_title("Learning curve for decision tree")
_ = plt.legend()

# %% [markdown]
# On this learning curve, we see that more samples we add in the training set,
# lower the error becomes. With this curve, we are searching for the "plateau",
# for which there is not benefit to add anymore samples or address the
# potential gain of adding more sample into the training set.
#
# ## Comparing results with baseline and chance level
# Previously, we compare the generalization error by taking into account the
# target distribution. A good practice is to compare the generalization error
# with a dummy baseline and the chance level. In regression, we could use the
# `DummyRegressor` and predict the mean target without using the data. The
# chance level can be determined by permuting the labels and check the
# difference of result.

# %%
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor()
result_dummy = cross_validate(
    dummy, X, y, cv=cv, scoring="neg_mean_absolute_error",
)

# %%
from sklearn.model_selection import permutation_test_score

score, permutation_score, pvalue = permutation_test_score(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
    n_jobs=-1, n_permutations=30,
)

# %% [markdown]
# We plot the generalization errors for each of the experiment. We see that
# even our regressor does not perform well, it is far above chances our a
# regressor that would predict the mean target.

# %%
final_result = pd.concat(
    [
        result_cv["test_score"] * -1,
        pd.Series(result_dummy["test_score"]) * -1,
        pd.Series(permutation_score) * -1,
    ], axis=1
).rename(columns={
    "test_score": "Cross-validation score",
    0: "Dummy score",
    1: "Permuted score",
})

# %%
sns.displot(final_result, kind="kde")
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# ## Choice of cross-validation
