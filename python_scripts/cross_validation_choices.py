# %% [markdown]
# # Choice of cross-validation
#
# In the previous section, we presented the cross-validation framework.
# However, we always use the `ShuffleSplit` strategy to repeat the split. One
# should question if this approach is always the best option and that some
# other cross-validation strategies would be better adapted. Indeed, we will
# focus on three aspects that influenced the choice of the cross-validation
# strategy: class stratification, sample grouping, feature dependence.
#
# ## Stratification
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

model = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# Once we created our model, we will use the cross-validation framework to
# evaluate it. We will use a strategy called `KFold` cross-validation. We give
# a simple usage example of the `KFold` strategy to get an intuition of how it
# splits the dataset. We will define a dataset with nine samples and repeat the
# cross-validation three times (i.e. `n_splits`).

# %%
import numpy as np
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
from sklearn.model_selection import cross_validate

cv = KFold(n_splits=3)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %% [markdown]
# It is a real surprise that our model cannot correctly classify any sample in
# any cross-validation split. We will now check our target's value to
# understand the issue while we should have started with this step.

# %%
import seaborn as sns
sns.set_context("talk")

ax = y.plot()
ax.set_xlabel("Sample index")
ax.set_ylabel("Class")
ax.set_yticks(y.unique())
_ = ax.set_title("Class value in target y")

# %% [markdown]
# We see that the target vector `y` is ordered. It will have some unexpected
# consequences when using the `KFold` cross-validation. To illustrate the
# consequences, we will show the class count in each fold of the
# cross-validation in the train and test set.
#
# For this matter, we create a function, because we will reuse it, which given
# a cross-validation object and the data `X` and `y`, is returning a dataframe
# with the class counts by folds and by split sets.

# %%
from collections import Counter
import pandas as pd


def compute_class_count_cv(cv, X, y):
    class_probability = []
    for cv_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        # Compute the class probability for the training set
        train_class = Counter(y[train_index])
        class_probability += [
            ["Train set", f"CV #{cv_idx}", klass, proportion]
            for klass, proportion in train_class.items()
        ]
        # Compute the class probability for the test set
        test_class = Counter(y[test_index])
        class_probability += [
            ["Test set", f"CV #{cv_idx}", klass, proportion]
            for klass, proportion in test_class.items()
        ]

    class_probability = pd.DataFrame(
        class_probability, columns=["Set", "CV", "Class", "Count"])
    return class_probability


# %% [markdown]
# Let's compute the statistics using the `KFold` cross-validation and we will
# plot these information in a bar plot.

# %%
kfold_class_count = compute_class_count_cv(cv, X, y)
kfold_class_count

# %%
g = sns.FacetGrid(kfold_class_count, col="Set")
g.map_dataframe(
    sns.barplot, x="Class", y="Count", hue="CV", palette="tab10")
g.set_axis_labels("Class", "Count")
g.add_legend()
_ = g.fig.suptitle("Class count with K-fold cross-validation", y=1.05)

# %% [markdown]
# We can confirm that in each fold, only two of the three classes are present
# in the training set and all samples of the remaining class is used as a test
# set. So our model is unable to predict this class that was unseen during the
# training stage.
#
# One possibility to solve the issue is to shuffle the data before to split the
# data into three groups.

# %%
cv = KFold(n_splits=3, shuffle=True, random_state=0)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %% [markdown]
# We get results that are closer to what we would expect with an accuracy above
# 90%. Now that we solved our first issue, it would be interesting to check if
# the class frequency in the training and testing set is equal to our original
# set's class frequency. It would ensure that we are training and testing our
# model with a class distribution that we will encounter in production.

# %%
kfold_shuffled_class_count = compute_class_count_cv(cv, X, y)

g = sns.FacetGrid(kfold_shuffled_class_count, col="Set")
g.map_dataframe(
    sns.barplot, x="Class", y="Count", hue="CV", palette="tab10")
g.set_axis_labels("Class", "Count")
g.add_legend()
_ = g.fig.suptitle(
    "Class count with shuffled K-fold cross-validation", y=1.05)

# %% [markdown]
# We see that neither the training and testing sets have the same class
# frequencies as our original dataset because the count for each class is
# varying a little.
#
# However, one might want to split our data by preserving the original class
# frequencies: we want to **stratify** our data by class. In scikit-learn, some
# cross-validation strategies are implementing the stratification and contains
# `Stratified` in their names.

# %%
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)

# %%
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %%
stratified_kfold_class_count = compute_class_count_cv(cv, X, y)

g = sns.FacetGrid(stratified_kfold_class_count, col="Set")
g.map_dataframe(
    sns.barplot, x="Class", y="Count", hue="CV", palette="tab10")
g.set_axis_labels("Class", "Count")
g.add_legend()
_ = g.fig.suptitle(
    "Class count with stratifiedK-fold cross-validation", y=1.05)

# %% [markdown]
# In this case, we observe that the class counts either in the train set or the
# test set are very close. The difference is due to the small number of samples
# in the iris dataset.
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
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split

X, y = quotes.drop(columns=["Chevron"]), quotes["Chevron"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0,
)

# %% [markdown]
# We will use a decision tree regressor that we expect to overfit and thus not
# generalize to unseen data. We will use a `ShuffleSplit` cross-validation to
# check the performance of our model.

# %%
from sklearn.model_selection import ShuffleSplit
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
# We obtain similar good results in terms of $R^2$. We will plot the
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
