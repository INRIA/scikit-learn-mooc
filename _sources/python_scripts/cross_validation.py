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
# # Evaluation of your predictive model: the cross-validation framework.

# %% [markdown]
# ## Introduction
# In the previous notebooks, we check how to fit a machine-learning model. When
# we evaluate our model's performance, we did not detail the evaluation
# framework that one should use in machine-learning. This notebook presents the
# cross-validation framework and emphasizes the importance of evaluating a
# model with such a framework.
#
# Besides, we will show some good practices to follow, such as nested
# cross-validation, when tuning model parameters.

# %% [markdown]
# ## Train and test datasets
# Before discussing the cross-validation framework, we will linger on the
# reasons for always having training and testing sets. Let's first look at the
# limitation of using a unique dataset.
#
# ### Load the California housing dataset

# %%
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

# %% [markdown]
# We use this dataset to predict the median value of houses in an area in
# California. The feature collected are based on general real-estate
# and geographical information.

# %%
print(housing.DESCR)

# %%
X.head()

# %% [markdown]
# To simplify future visualization, we transform the target in k\$.

# %%
y *= 100
y.head()

# %% [markdown]
# ### Empirical error vs generalization error
#
# As mentioned previously, we start by fitting a decision tree regressor on the
# full dataset.

# %%
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# %% [markdown]
# After training the regressor, we would like to know the regressor's potential
# performance once deployed in production. For this purpose, we use the mean
# absolute error, which gives us an error in the native unit, i.e. k\$.

# %%
from sklearn.metrics import mean_absolute_error

y_pred = regressor.predict(X)
score = mean_absolute_error(y_pred, y)
print(f"In average, our regressor make an error of {score:.2f} k$")

# %% [markdown]
# A perfect prediction is with no error is too optimistic and almost always
# revealing a methodological problem when doing machine learning.
#
# Indeed, we trained and predicted on the same dataset. Since our decision tree
# was fully grown, every sample in the dataset is stored in a leaf node.
# Therefore, our decision tree fully memorized the dataset given during `fit`
# and make no single error when predicting on the same data.
#
# This error computed above is called the **empirical error** or **training
# error**.
#
# We trained a predictive model to minimize the empirical error but our aim is
# to minimize the error on data that has not been seen during training.
#
# This error is also called the **generalization error** or the "true"
# **testing error**. Thus, the most basic evaluation involves:
#
# * splitting our dataset into two subsets: a training set and a testing set;
# * fitting the model on the training set;
# * estimating the empirical error on the training set;
# * estimating the generalization error on the testing set.

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
print(f"The empirical error of our model is {score:.2f} k$")

# %%
y_pred = regressor.predict(X_test)
score = mean_absolute_error(y_pred, y_test)
print(f"The generalization error of our model is {score:.2f} k$")

# %% [markdown]
# ### Stability of the cross-validation estimates
#
# When doing a single train-test split we don't give any indication
# regarding the robustness of the evaluation of our predictive model: in
# particular, if the test set is small, this estimate of the generalization
# error can be unstable and do not reflect the "true error rate" we would have
# observed with the same model on an unlimitted amount of test data.
#
# For instance, we could have been lucky when we did our random split of our
# limited dataset and isolated some of the easiest cases to predict in the
# testing set just by chance: the estimation of the generalization error would
# be overly optimistic, in this case.
#
# **Cross-validation** allows estimating the robustness of a predictive model
# by repeating the splitting procedure. It will give several empirical and
# generalization errors and thus some **estimate of the variability of the
# model performance**.
#
# There are different cross-validation strategies, for now we are going to
# focus on one called "shuffle-split". At each iteration of this strategy we:
#
# - shuffle the order of the samples of a copy of the full data at random;
# - split the shuffled dataset into a train and a test set;
# - train a new model on the train set;
# - evaluate the generalization error on the test set.
#
# We repeat this procedure `n_splits` times. Using `n_splits=30` means that we
# will train 30 models in total and all of them will be discarded: we just
# record their performance on each variant of the test set.
#
# To evaluate the performance of our regressor, we can use `cross_validate`
# with a `ShuffleSplit` object:

# %%
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)

cv_results = cross_validate(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
)
cv_results = pd.DataFrame(cv_results)

# %% [markdown]
# By convention, scikit-learn model evaluation tools always use a convention
# where "higher is better", this explains we used
# `scoring="neg_mean_absolute_error"` (meaning "negative mean absolute error").
#
# Let us revert the negation to get the actual error and not the negative
# score:

# %%
cv_results["test_error"] = -cv_results["test_score"]

# %% [markdown]
# Let's check the results reported by the cross-validation.

# %%
cv_results.head(10)

# %% [markdown]
# We get timing information to fit and predict at each round of
# cross-validation. Also, we get the test score, which corresponds to the
# generalization error on each of the split.

# %%
len(cv_results)

# %% [markdown]
# We get 30 entries in our resulting dataframe because we performed 30 splits.
# Therefore, we can show the generalization error distribution and thus, have
# an estimate of its variability.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(cv_results["test_error"], kde=True, bins=10)
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# We observe that the generalization error is clustered around 45.5 k\\$ and
# ranges from 43 k\\$ to 49 k\\$.

# %%
print(
    f"The mean cross-validated generalization error is: "
    f"{cv_results['test_error'].mean():.2f} k$"
)

# %%
print(
    f"The standard deviation of the generalization error is: "
    f"{cv_results['test_error'].std():.2f} k$"
)
# %% [markdown]
# Note that the standard deviation is much smaller than the mean:
# we could summarize that our cross-validation estimate of the
# generalization error is 45.7 +/- 1.1 k\\$.
#
# If we were to train a single model on the full dataset (without
# cross-validation) and then had later access to an unlimited
# amount of test data, we would expect its true generalization
# error to fall close to that region.
#
# While this information is interesting in it-self, this should be
# contrasted to the scale of the natural variability of the target `y`
# in our dataset.
#
# Let us plot the distribution of the target variable:

# %%
sns.displot(y, kde=True, bins=20)
_ = plt.xlabel("Median House Value (k$)")

# %%
print(f"The standard deviation of the target is: {y.std():.2f} k$")

# %% [markdown]
# The target variable ranges from close to 0 k\\$ up to 500 k\\$ and, with a
# standard deviation around 115 k\\$.
#
# We notice that the mean estimate of the generalization error obtained
# by cross-validation is a bit than the natural scale of variation
# of the target variable. Furthermore the standard deviation of the cross
# validation estimate of the generalization error is even much smaller.
#
# This is a good start, but not necessarily enough to decide whether the
# generalization performance is good enough to make our prediction useful in
# practice.
#
# We recall that our model makes, on average, an error around 45 k\$. With this
# information and looking at the target distribution, such an error might be
# acceptable when predicting houses with a 500 k\\$. However, it would be an
# issue with a house with a value of 50 k\\$. Thus, this indicates that our
# metric (Mean Absolute Error) is not ideal.
#
# We might instead choose a metric relative to the target
# value to predict: the mean absolute percentage error would have been a much
# better choice.
#
# But in all cases, an error of 45 k\$ might be too large to automatically use
# our model to tag house value without expert supervision.
#
# To better understand the performance of our model and maybe find insights
# on how to improve it we will compare the generalization
# error with the empirical error. Thus, we need to compute the error on the
# training set, which is possible using the `cross_validate` function.

# %%
cv_results = cross_validate(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
    return_train_score=True,
    n_jobs=2
)
cv_results = pd.DataFrame(cv_results)

# %%
scores = pd.DataFrame()
scores[["train_error", "test_score"]] = -cv_results[["train_score", "test_score"]]
sns.histplot(scores, bins=50)
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]

# By plotting the distribution of the empirical and generalization errors, we
# get information about whether our model is over-fitting, under-fitting (or
# both at the same time).
#
# Here, we observe a **small empirical error** (actually zero), meaning that
# the model is **not under-fitting**: it is flexible enough to capture any
# variations present in the training set.
#
# However the **significantly larger generalization error** tells us that the
# model is **over-fitting**: the model has memorized many variations of the
# training set that could be considered "noisy" because they do not generalize
# to help us make good prediction on the test set.
#
# Some model hyper-parameters are usually the key to go from a model that
# underfits to a model that overfits, hopefully going through a region were we
# can get a good balance between the two. We can acquire knowledge by plotting
# a curve called the validation curve. This curve applies the above experiment
# and varies the value of a hyper-parameter.
#
# For the decision tree, the `max_depth` the main parameter to control the
# trade-off between under-fitting and over-fitting.

# %%
# %%time
from sklearn.model_selection import validation_curve

max_depth = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    regressor, X, y,
    param_name="max_depth", param_range=max_depth,
    cv=cv, scoring="neg_mean_absolute_error",
    n_jobs=2,
)

# %%
_, ax = plt.subplots()
ax.plot(
    max_depth, -train_scores.mean(axis=1),
    linestyle="-.", label="empirical error",
    alpha=0.8,
)
ax.fill_between(
    max_depth,
    -train_scores.mean(axis=1) - train_scores.std(axis=1),
    -train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.5,
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
)

ax.set_xticks(max_depth)
ax.set_xlabel("Maximum depth of decision tree")
ax.set_ylabel("Mean absolute error (k$)")
ax.set_title("Validation curve for decision tree")
_ = plt.legend()

# %% [markdown]
# The validation curve can be divided into three areas:
#
# - For `max_depth < 10`, the decision tree underfits. The empirical error and
#   therefore also the generalization error are both high. The model is too
#   constrained and cannot capture much of the varibility of the target
#   variable.
#
#
# - The region around `max_depth = 10` corresponds to the parameter for which
#   the decision tree generalizes the best. It is flexible enough to capture a
#   fraction of the variability of the target that generalizes, while not
#   memorizing all of the noise in the target.
#
#
# - For `max_depth > 10`, the decision tree overfits. The empirical error
#   becomes very small, while the generalization error increases. In this
#   region, the models captures too much of the noisy part of the variations of
#   the target and this harms its ability to generalize well to test data.
#
#
# Note that for `max_depth = 10`, the model overfits a bit as there is a gap
# between the empirical error and the generalization error. It can also
# potentially underfit also a bit at the same time, because the empirical error
# is still far from zero (more than 30 k\\$), meaning that the model might
# still be too constrained to model interesting parts of the data. However the
# generalization error is minimal, and this is what really matters. This is the
# best compromise we could reach by just tuning this parameter.
#
# We were lucky that the variance of the errors was small compared to their
# respective values, and therefore the conclusions above are quite clear. This
# is not necessarily always the case.
#
# We will now focus on one factor that can affect this variance, namely, the
# size of the dataset:

# %%
y.size

# %% [markdown]
# Let's do an experiment and reduce the number of samples and repeat the
# previous experiment.


# %%
def make_cv_analysis(regressor, X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    cv_results = pd.DataFrame(
        cross_validate(
            regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
            return_train_score=True
        )
    )
    return (cv_results["test_score"] * -1).values

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
# For the different sample sizes, we plotted the distribution of the
# generalization error. We observe that smaller is the sample size; larger is
# the variance of the generalization errors. Thus, having a small number of
# samples might put us in a situation where it is impossible to get a reliable
# evaluation.
#
# Here, we plotted the different curve to highlight the issue of small sample
# size. However, this experiment is also used to draw the so-called **learning
# curve**. This curve gives some additional indication regarding the benefit of
# adding new training samples to improve a model's performance.

# %%
from sklearn.model_selection import learning_curve

results = learning_curve(
    regressor,
    X,
    y,
    train_sizes=sample_sizes[:-1],
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=2,
)
train_size, train_scores, test_scores = results[:3]
train_errors, test_errors = -train_scores, -test_scores

# %%
_, ax = plt.subplots()
ax.plot(
    train_size,
    train_errors.mean(axis=1),
    linestyle="-.",
    label="empirical error",
    alpha=0.8,
)
ax.fill_between(
    train_size,
    train_errors.mean(axis=1) - train_errors.std(axis=1),
    train_errors.mean(axis=1) + train_errors.std(axis=1),
    alpha=0.5,
)
ax.plot(
    train_size,
    test_errors.mean(axis=1),
    linestyle="-.",
    label="Generalization error",
    alpha=0.8,
)
ax.fill_between(
    train_size,
    test_errors.mean(axis=1) - test_errors.std(axis=1),
    test_errors.mean(axis=1) + test_errors.std(axis=1),
    alpha=0.5,
)

ax.set_xticks(train_size)
ax.set_xscale("log")
ax.set_xlabel("Number of samples in the training set")
ax.set_ylabel("Mean absolute error (k$)")
ax.set_title("Learning curve for decision tree")
_ = plt.legend()

# %% [markdown]
# We see that the more samples we add to the training set on this learning
# curve, the lower the error becomes. With this curve, we are searching for the
# plateau for which there is no benefit to adding samples anymore or assessing
# the potential gain of adding more samples into the training set.
#
# For this dataset we notice that our decision tree model would really benefit
# from additional datapoints to reduce the amount of over-fitting and hopefully
# reduce the generalization error even further.
#
#
# ## Comparing results with baseline and chance level
#
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
    dummy,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
)

# %%
from sklearn.model_selection import permutation_test_score

score, permutation_score, pvalue = permutation_test_score(
    regressor,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    n_permutations=30,
)

# %% [markdown]
# We plot the generalization errors for each of the experiments. Even if our
# regressor does not perform well, it is far above a regressor that would
# predict the mean target.

# %%
final_result = pd.concat(
    [
        cv_results["test_score"] * -1,
        pd.Series(result_dummy["test_score"]) * -1,
        pd.Series(permutation_score) * -1,
    ],
    axis=1,
).rename(
    columns={
        "test_score": "Cross-validation score",
        0: "Dummy score",
        1: "Permuted score",
    }
)

# %%
sns.displot(final_result, kind="kde")
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# ## Choice of cross-validation
# In the previous section, we presented the cross-validation framework.
# However, we always use the `ShuffleSplit` strategy to repeat the split. One
# should question if this approach is always the best option and that some
# other cross-validation strategies would be better adapted. Indeed, we will
# focus on three aspects that influenced the choice of the cross-validation
# strategy: class stratification, sample grouping, feature dependence.
#
# ### Stratification
# Let's start with the concept of stratification by giving an example where
# we can get into trouble if we are not careful. We load the iris dataset.

# %%
from sklearn.datasets import load_iris

X, y = load_iris(as_frame=True, return_X_y=True)

# %% [markdown]
# At this point, we create a basic machine-learning model: a logistic
# regression. We expect this model to work quite well on the iris dataset since
# this is a toy dataset.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
)

# %% [markdown]
# Once we created our model, we will use the cross-validation framework to
# evaluate it. We will use a strategy called `KFold` cross-validation. We give
# a simple usage example of the `KFold` strategy to get an intuition of how it
# splits the dataset. We will define a dataset with nine samples and repeat the
# cross-validation three times (i.e. `n_splits`).

# %%
from sklearn.model_selection import KFold

X_random = np.random.randn(9, 1)
cv = KFold(n_splits=3)
for train_index, test_index in cv.split(X_random):
    print("TRAIN:", train_index, "TEST:", test_index)

# %% [markdown]
# By defining three splits, we will use three samples each time for testing and
# 6 for training. `KFold` does not shuffle by default. It means that it will
# select the three first samples for the testing set at the first split, then
# the three next three samples for the second split, and the three next for the
# last split. In the end, all samples have been used in testing at least once
# among the different splits.
#
# Now, let's apply this strategy to check the performance of our model.

# %%
cv = KFold(n_splits=3)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is {test_score.mean()}")

# %% [markdown]
# It is a real surprise that our model cannot correctly classify any sample in
# any cross-validation split. We will now check our target's value to
# understand the issue while we should have started with this step.

# %%
y.tolist()

# %%
y.value_counts(normalize=True)

# %% [markdown]
# By looking at our target, samples of a class are grouped together. Also, the
# sample count per class is the same. Thus, splitting the data with three
# splits use all samples of a single class during testing. So our model is
# unable to predict this class that was unseen during the training stage.
#
# One possibility to solve the issue is to shuffle the data before to split the
# data into three groups.

# %%
cv = KFold(n_splits=3, shuffle=True, random_state=0)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is {test_score.mean():.3f}")

# %% [markdown]
# We get results that are closer to what we would expect with an accuracy above
# 90%. Now that we solved our first issue, it would be interesting to check if
# the class frequency in the training and testing set is equal to our original
# set's class frequency. It would ensure that we are training and testing our
# model with a class distribution that we will encounter in production.

# %%
for train_index, test_index in cv.split(X, y):
    print(
        f"Class frequency in the training set:\n"
        f"{y[train_index].value_counts(normalize=True).sort_index()}"
    )
    print(
        f"Class frequency in the testing set:\n"
        f"{y[test_index].value_counts(normalize=True).sort_index()}"
    )

# %% [markdown]
# We see that neither the training and testing sets have the same class
# frequencies as our original dataset. Thus, it means that one might want to
# split our data by preserving the original class frequencies: we want to
# **stratify** our data by class. In scikit-learn, some cross-validation
# strategies are implementing the stratification and contains `Stratified` in
# their names.

# %%
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)
for train_index, test_index in cv.split(X, y):
    print(
        f"Class frequency in the training set:\n"
        f"{y[train_index].value_counts(normalize=True).sort_index()}"
    )
    print(
        f"Class frequency in the testing set:\n"
        f"{y[test_index].value_counts(normalize=True).sort_index()}"
    )

# %%
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is {test_score.mean():.3f}")

# %% [markdown]
# In this case, we observe that the class frequencies are very close. The
# difference is due to the small number of samples in the iris dataset.
#
# In conclusion, this is a good practice to use stratification within the
# cross-validation framework when dealing with a classification problem.
#
# ## Sample grouping
# We are going to linger into the concept of samples group. As in the previous
# section, we will give an example to highlight some surprising results. This
# time, we will use the handwritten digits dataset.

# %%
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# %% [markdown]
# We will use the same baseline model. We will use a `KFold` cross-validation
# without shuffling the data at first.

# %%
cv = KFold(shuffle=False)
results = cross_validate(model, X, y, cv=cv, n_jobs=-1)
test_score_no_shuffling = results["test_score"]
print(f"The average accuracy is {test_score_no_shuffling.mean():.3f}")

# %%
cv = KFold(shuffle=True)
results = cross_validate(model, X, y, cv=cv, n_jobs=-1)
test_score_with_shuffling = results["test_score"]
print(f"The average accuracy is {test_score_with_shuffling.mean():.3f}")

# %% [markdown]
# We observe that shuffling the data allows to improving the mean accuracy.
# We could go a little further and plot the distribution of the generalization
# score.

# %%
sns.displot(data=pd.DataFrame(
    [test_score_no_shuffling, test_score_with_shuffling],
    index=["KFold without shuffling", "KFold with shuffling"],
).T, kde=True, bins=10)
plt.xlim([0.8, 1.0])
plt.xlabel("Accuracy score")

# %% [markdown]
# The cross-validation generalization error that uses the shuffling has less
# variance than the one that does not impose any shuffling. It means that some
# specific fold leads to a low score in the unshuffle case.

# %%
print(test_score_no_shuffling)

# %% [markdown]
# Thus, there is an underlying structure in the data that shuffling will break
# and get better results. To get a better understanding, we should read the
# documentation shipped with the dataset.

# %%
print(digits.DESCR)

# %% [markdown]
# If we read carefully, 13 writers wrote the digits of our dataset, accounting
# for a total amount of 1797 samples. Thus, a writer wrote several times the
# same numbers. Let's suppose that the writer samples are grouped.
# Subsequently, not shuffling the data will keep all writer samples together
# either in the training or the testing sets. Mixing the data will break this
# structure, and therefore digits written by the same writer will be available
# in both the training and testing sets.
#
# Besides, a writer will usually tend to write digits in the same manner. Thus,
# our model will learn to identify a writer's pattern for each digit instead of
# recognizing the digit itself.
#
# We can solve this problem by ensuring that the data associated with a writer
# should either belong to the training or the testing set. Thus, we want to
# group samples for each writer.
#
# Here, we will manually define the group for the 13 writers.

# %%
from itertools import count

# defines the lower and upper bounds of sample indices
# for each writer
writer_boundaries = [
    0, 130, 256, 386, 516, 646, 776, 915, 1029,
    1157, 1287, 1415, 1545, 1667, 1797
]
groups = np.zeros_like(y)

for group_id, lower_bound, upper_bound in zip(
    count(),
    writer_boundaries[:-1],
    writer_boundaries[1:]
):
    groups[lower_bound:upper_bound] = group_id
groups

# %% [markdown]
# Once we group the digits by writer, we can use cross-validation to take this
# information into account: the class containing `Group` should be used.

# %%
from sklearn.model_selection import GroupKFold

cv = GroupKFold()
results = cross_validate(model, X, y, groups=groups, cv=cv, n_jobs=-1)
test_score = results["test_score"]
print(f"The average accuracy is {test_score.mean():.3f}")

# %% [markdown]
# We see that this strategy is less optimistic regarding the model performance.
# However, this is the most reliable if our goal is to make handwritten digits
# recognition writers independent.
#
# ## Non i.i.d. data
# In machine learning, it is quite common to assume that the data are i.i.d,
# meaning that the generative process does not have any memory of past samples
# to generate new samples.
#
# This assumption is usually violated when dealing with time series. A sample
# depends on past information.
#
# We will take an example to highlight such issues with non-i.i.d. data in the
# previous cross-validation strategies presented. We are going to load
# financial quotations from some energy companies.

# %%
symbols = {
    "TOT": "Total",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
}
url = (
    "https://raw.githubusercontent.com/scikit-learn/examples-data/"
    "master/financial-data/{}.csv"
)

quotes = {}
for symbol in symbols:
    data = pd.read_csv(url.format(symbol), index_col=0, parse_dates=True)
    quotes[symbols[symbol]] = data["open"]
quotes = pd.DataFrame(quotes)

# %% [markdown]
# We can start by plotting different financial quotations.

# %%
_, ax = plt.subplots(figsize=(10, 6))
_ = quotes.plot(ax=ax)

# %% [markdown]
# We can formulate the following regression problem. We want to be able to
# predict the quotation of Chevron using all other energy companies' quotes.

# %%
X, y = quotes.drop(columns=["Chevron"]), quotes["Chevron"]

# %% [markdown]
# We will use a decision tree regressor that we expect to overfit and thus not
# generalize to unseen data. We will use a `ShuffleSplit` cross-validation to
# check the performance of our model.

# %%
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

cv = ShuffleSplit(random_state=0)
results = cross_validate(regressor, X_train, y_train, cv=cv, n_jobs=-1)
test_score = results["test_score"]
print(f"The mean R2 is: {test_score.mean():.2f}")

# %% [markdown]
# Surprisingly, we get outstanding performance. We will investigate and find
# the reason for such good results with a model that is expected to fail. We
# previously mentioned that `ShuffleSplit` is an iterative cross-validation
# scheme that shuffles data and split. We will simplify this procedure with a
# single split and plot the prediction. We can use `train_test_split` for this
# purpose.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0,
)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# Affect the index of `y_test` to ease the plotting
y_pred = pd.Series(y_pred, index=y_test.index)

# %% [markdown]
# Let's check the performance of our model on this split.

# %%
from sklearn.metrics import r2_score

test_score = r2_score(y_test, y_pred)
print(f"The R2 on this single split is: {test_score:.2f}")

# %% [markdown]
# We obtain similar good results in terms of :math`R^2`. We will plot the
# training, testing and prediction samples.

# %%
_, ax = plt.subplots(figsize=(10, 8))
y_train.plot(ax=ax, label="Training")
y_test.plot(ax=ax, label="Testing")
y_pred.plot(ax=ax, label="Prediction")
_ = plt.legend()

# %% [markdown]
# So in this context, it seems that the model predictions are following the
# testing. But we can as well see that the testing samples are next to some
# training sample. And with these time-series, we see a relationship between a
# sample at the time `t` and a sample at `t+1`. In this case, we are violating
# the i.i.d. assumption. The insight to get is the following: a model can
# output of its training set at the time `t` for a testing sample at the time
# `t+1`. This prediction would be closed to the true value even if our model
# did not learn anything else than memorizing the training dataset.
#
# An easy way to verify this hypothesis is not to shuffle the data when doing
# the split. In this case, we will use the first 75% of the data to train and
# the remaining data to test.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, random_state=0,
)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)

# %%
test_score = r2_score(y_test, y_pred)
print(f"The R2 on this single split is: {test_score:.2f}")

# %% [markdown]
# In this case, we see that our model is not magical anymore. Indeed, it
# performs worse than just predicting the mean of the target. We can visually
# check what we are predicting.

# %%
_, ax = plt.subplots(figsize=(10, 8))
y_train.plot(ax=ax, label="Training")
y_test.plot(ax=ax, label="Testing")
y_pred.plot(ax=ax, label="Prediction")
_ = plt.legend()

# %% [markdown]
# We see that our model cannot predict anything because it doesn't have samples
# around the testing sample. Let's check how we could have made a proper
# cross-validation scheme to get a reasonable performance estimate.
#
# One solution would be to group the samples into time blocks, e.g. by quarter,
# and predict each group's information by using information from the other
# groups. We can use the `LeaveOneGroupOut` cross-validation for this purpose.

# %%
from sklearn.model_selection import LeaveOneGroupOut

groups = quotes.index.to_period("Q")
cv = LeaveOneGroupOut()
results = cross_validate(
    regressor, X, y, cv=cv, groups=groups,
    n_jobs=-1
)
test_score = results["test_score"]
print(f"The mean R2 is: {test_score.mean():.2f}")

# %% [markdown]
# In this case, we see that we cannot make good predictions, which is less
# surprising than our original results.
#
# Another thing to consider is the actual application of our solution. If our
# model is aimed at forecasting (i.e., predicting future data from past data),
# we should not use training data that are ulterior to the testing data. In
# this case, we can use the `TimeSeriesSplit` cross-validation to enforce this
# behaviour.

# %%
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=groups.nunique())
results = cross_validate(
    regressor, X, y, cv=cv, groups=groups,
    n_jobs=-1
)
test_score = results["test_score"]
print(f"The mean R2 is: {test_score.mean():.2f}")

# %% [markdown]
# ## Nested cross-validation
# Cross-validation is a powerful tool to evaluate the performance of a model.
# It is also used to select the best model from a pool of models. This pool of
# models can be the same family of predictor but with different parameters. In
# this case, we call this procedure **fine-tuning** of the model
# hyperparameters.
#
# We could also imagine that we would like to choose among heterogeneous models
# that will similarly use the cross-validation.
#
# In the example below, we show a minimal example of using the utility
# `GridSearchCV` to find the best parameters via cross-validation.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [.01, .1],
}

model = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=KFold(),
    n_jobs=-1,
)
model.fit(X, y)

# %% [markdown]
# We recall that `GridSearchCV` will train a model with some specific parameter
# on a training set and evaluate it on testing. However, this evaluation is
# done via cross-validation using the `cv` parameter. This procedure is
# repeated for all possible combinations of parameters given in `param_grid`.
#
# The attribute `best_params_` will give us the best set of parameters that
# maximize the mean score on the internal test sets.

# %%
print(f"The best parameter found are: {model.best_params_}")

# %% [markdown]
# We can now show the mean score obtained using the parameter `best_score_`.

# %%
print(f"The mean score in CV is: {model.best_score_:.3f}")

# %% [markdown]
# At this stage, one should be extremely careful using this score. The
# misinterpretation would be the following: since the score was computed on a
# test set, it could be considered our model's generalization score.
#
# However, we should not forget that we used this score to pick-up the best
# model. It means that we used knowledge from the test set (i.e. test score) to
# decide our model's training parameter.
#
# Thus, this score is not a reasonable estimate of our generalization error.
# Indeed, we can show that it will be too optimistic in practice. The good way
# is to use a "nested" cross-validation. We will use an inner cross-validation
# corresponding to the previous procedure shown to optimize the
# hyper-parameters. We will also include this procedure within an outer
# cross-validation, which will be used to estimate the generalization error of
# our fine-tuned model.
#
# In this case, our inner cross-validation will always get the training set of
# the outer cross-validation, making it possible to compute the generalization
# score on a completely independent set.
#
# We will show below how we can create such nested cross-validation and obtain
# the generalization score.

# %%
# Declare the inner and outer cross-validation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)

# Inner cross-validation for parameter search
model = GridSearchCV(
    estimator=SVC(), param_grid=param_grid, cv=inner_cv,
    n_jobs=-1,
)

# Outer cross-validation to compute the generalization score
result = cross_validate(
    model, X, y, cv=outer_cv, n_jobs=-1,
)
test_score = result["test_score"].mean()
print(
    f"The mean score using nested cross-validation is: "
    f"{test_score.mean():.3f}"
)

# %% [markdown]
# In the example above, the reported score is more trustful and should be close
# to production's expected performance.
#
# We will illustrate the difference between the nested and non-nested
# cross-validation scores to show that the latter one will be too optimistic in
# practice.

# %%
test_score_not_nested = []
test_score_nested = []

N_TRIALS = 20
for i in range(N_TRIALS):
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    model = GridSearchCV(
        estimator=SVC(), param_grid=param_grid, cv=inner_cv,
        n_jobs=-1,
    )
    model.fit(X, y)
    test_score_not_nested.append(model.best_score_)

    # Nested CV with parameter optimization
    result = cross_validate(
        model, X, y, cv=outer_cv, n_jobs=-1,
    )
    test_score_nested.append(result["test_score"].mean())

# %%
df = pd.DataFrame(
    {
        "Not nested CV": test_score_not_nested,
        "Nested CV": test_score_nested,
    }
)
ax = df.plot(kind="box")
ax.set_ylabel("Accuracy")
_ = ax.set_title(
    "Comparison of mean accuracy obtained on the test sets with\n"
    "and without nested cross-validation"
)

# %% [markdown]
# We observe that the model's performance with the nested cross-validation is
# not as good as the non-nested cross-validation.
#
# ## Take away
# In this notebook, we presented the framework used in machine-learning to
# evaluate a predictive model's performance: the cross-validation.
#
# Besides, we presented several splitting strategies that can be used in the
# general cross-validation framework. These strategies should be used wisely
# when encountering some specific patterns or types of data.
#
# Finally, we show how to perform nested cross-validation to select an optimal
# model and evaluate its generalization performance.
