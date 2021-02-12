# %% [markdown]
# # The framework and why do we need it
#
# In this notebook, we present the general cross-validation framework. Before
# we dive in, let's linger on the reasons for always having training
# and testing sets. Let's first look at the limitation of using a dataset
# without keeping any samples out.
#
# To illustrate the different concepts, we will use the California housing
# dataset.

# %%
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

# %% [markdown]
# We recall that in this dataset, the aim is to predict the median value of
# houses in an area in California. The features collected are based on general
# real-estate and geographical information.

# %%
print(housing.DESCR)

# %%
X.head()

# %% [markdown]
# To simplify future visualization, let's transform the prices from the
# dollar ($) range to the thousand dollars (k$) range.

# %%
y *= 100
y.head()

# %% [markdown]
# ## Empirical error vs generalization error
#
# As we are trying to predict prices, a quantitative feature, this is a
# regression problem. Therefore, we will fit a decision tree regressor
# on the entire dataset.

# %%
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# %% [markdown]
# After training the regressor, we would like to know its potential
# performance once deployed in production. For this purpose, we use the mean
# absolute error, which gives us an error in the native unit, i.e. k\$.

# %%
from sklearn.metrics import mean_absolute_error

y_pred = regressor.predict(X)
score = mean_absolute_error(y_pred, y)
print(f"On average, our regressor makes an error of {score:.2f} k$")

# %% [markdown]
# We get perfect prediction with no error. It is too optimistic and almost
# always revealing a methodological problem when doing machine learning.
#
# Indeed, we trained and predicted on the same dataset. Since our decision tree
# was fully grown, every sample in the dataset is stored in a leaf node.
# Therefore, our decision tree fully memorized the dataset given during `fit`
# and therefore made no error when predicting.
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
#
# So let's split our dataset.
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
# Then, let's train our model

# %%
regressor.fit(X_train, y_train)

# %% [markdown]
# Finally, we can estimate the different types of errors. Let's start by
# computing the empirical error.

# %%
y_pred = regressor.predict(X_train)
score = mean_absolute_error(y_pred, y_train)
print(f"The empirical error of our model is {score:.2f} k$")

# %% [markdown]
# We observe the same phenomena as in the previous experiment:
# our model memorized the training set.
# However, we can now compute the generalization error on the testing set.

# %%
y_pred = regressor.predict(X_test)
score = mean_absolute_error(y_pred, y_test)
print(f"The generalization error of our model is {score:.2f} k$")

# %% [markdown]
# This generalization error is actually about what we would expect from
# our model if it was used in a production environment.

# %% [markdown]
# ## Stability of the cross-validation estimates
#
# When doing a single train-test split we don't give any indication
# regarding the robustness of the evaluation of our predictive model: in
# particular, if the test set is small, this estimate of the generalization
# error can be unstable and wouldn't reflect the "true error rate" we would have
# observed with the same model on an unlimited amount of test data.
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
# - randomly shuffle the order of the samples of a copy of the full dataset;
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
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)
cv_results = cross_validate(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error")

# %% [markdown]
# The results `cv_results` are stored into a Python dictionary. We will convert
# it into a pandas dataframe to ease visualization and manipulation.

# %%
import pandas as pd

cv_results = pd.DataFrame(cv_results)
cv_results.head()

# %% [markdown]
# ```{tip}
# By convention, scikit-learn model evaluation tools always use a convention
# where "higher is better", this explains we used
# `scoring="neg_mean_absolute_error"` (meaning "negative mean absolute error").
# ```
#
# Let us revert the negation to get the actual error:

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

sns.set_context("talk")  # Set Seaborn's plotting style to "talk" mode.

sns.displot(cv_results["test_error"], kde=True, bins=10)
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# We observe that the generalization error is clustered around 45.5 k\$ and
# ranges from 43 k\$ to 49 k\$.

# %%
print(f"The mean cross-validated generalization error is: "
      f"{cv_results['test_error'].mean():.2f} k$")

# %%
print(f"The standard deviation of the generalization error is: "
      f"{cv_results['test_error'].std():.2f} k$")

# %% [markdown]
# Note that the standard deviation is much smaller than the mean: we could
# summarize that our cross-validation estimate of the generalization error is
# 45.7 +/- 1.1 k\$.
#
# If we were to train a single model on the full dataset (without
# cross-validation) and then had later access to an unlimited amount of test
# data, we would expect its true generalization error to fall close to that
# region.
#
# While this information is interesting in itself, it should be contrasted
# to the scale of the natural variability of the target `y` in our dataset.
#
# Let us plot the distribution of the target variable:

# %%
sns.displot(y, kde=True, bins=20)
_ = plt.xlabel("Median House Value (k$)")

# %%
print(f"The standard deviation of the target is: {y.std():.2f} k$")

# %% [markdown]
# The target variable ranges from close to 0 k\$ up to 500 k\$ and, with a
# standard deviation around 115 k\$.
#
# We notice that the mean estimate of the generalization error obtained by
# cross-validation is a bit smaller than the natural scale of variation of the
# target variable. Furthermore the standard deviation of the cross validation
# estimate of the generalization error is even smaller.
#
# This is a good start, but not necessarily enough to decide whether the
# generalization performance is good enough to make our prediction useful in
# practice.
#
# We recall that our model makes, on average, an error around 45 k\$. With this
# information and looking at the target distribution, such an error might be
# acceptable when predicting houses with a 500 k\$. However, it would be an
# issue with a house with a value of 50 k\$. Thus, this indicates that our
# metric (Mean Absolute Error) is not ideal.
#
# We might instead choose a metric relative to the target value to predict: the
# mean absolute percentage error would have been a much better choice.
#
# But in all cases, an error of 45 k\$ might be too large to automatically use
# our model to tag house value without expert supervision.
#
# ## Summary
#
# In this notebook, we saw:
#
# * the necessity of splitting the data into a train and test set;
# * the meaning of the empirical and generalization errors;
# * the overall cross-validation framework with the possibility to study
#   performance variations;
