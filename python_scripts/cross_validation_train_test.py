# %% [markdown]
# # Cross-validation framework
#
# In the previous notebooks, we introduce some concepts regarding the
# evaluation of predictive models. While this section could be slightly
# redundant, we intend to go into details into the cross-validation framework.
#
# Before we dive in, let's linger on the reasons for always having training and
# testing sets. Let's first look at the limitation of using a dataset without
# keeping any samples out.
#
# To illustrate the different concepts, we will use the California housing
# dataset.

# %%
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target

# %% [markdown]
# In this dataset, the aim is to predict the median value of houses in an area
# in California. The features collected are based on general real-estate and
# geographical information.
#
# Therefore, the task to solve is different from the one shown in the previous
# notebook. The target to be predicted is a continuous variable and not anymore
# discrete. This task is called regression.
#
# This, we will use a predictive model specific to regression and not to
# classification.

# %%
print(housing.DESCR)

# %%
data.head()

# %% [markdown]
# To simplify future visualization, let's transform the prices from the
# 100 (k\$) range to the thousand dollars (k\$) range.

# %%
target *= 100
target.head()

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# ## Training error vs testing error
#
# To solve this regression task, we will use a decision tree regressor.

# %%
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(data, target)

# %% [markdown]
# After training the regressor, we would like to know its potential generalization
# performance once deployed in production. For this purpose, we use the mean
# absolute error, which gives us an error in the native unit, i.e. k\$.

# %%
from sklearn.metrics import mean_absolute_error

target_predicted = regressor.predict(data)
score = mean_absolute_error(target, target_predicted)
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
# ```{note}
# In this MOOC, we will consistently use the term "training error".
# ```
#
# We trained a predictive model to minimize the training error but our aim is
# to minimize the error on data that has not been seen during training.
#
# This error is also called the **generalization error** or the "true"
# **testing error**.
#
# ```{note}
# In this MOOC, we will consistently use the term "testing error".
# ```
#
# Thus, the most basic evaluation involves:
#
# * splitting our dataset into two subsets: a training set and a testing set;
# * fitting the model on the training set;
# * estimating the training error on the training set;
# * estimating the testing error on the testing set.
#
# So let's split our dataset.
# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

# %% [markdown]
# Then, let's train our model.

# %%
regressor.fit(data_train, target_train)

# %% [markdown]
# Finally, we estimate the different types of errors. Let's start by computing
# the training error.

# %%
target_predicted = regressor.predict(data_train)
score = mean_absolute_error(target_train, target_predicted)
print(f"The training error of our model is {score:.2f} k$")

# %% [markdown]
# We observe the same phenomena as in the previous experiment: our model
# memorized the training set. However, we now compute the testing error.

# %%
target_predicted = regressor.predict(data_test)
score = mean_absolute_error(target_test, target_predicted)
print(f"The testing error of our model is {score:.2f} k$")

# %% [markdown]
# This testing error is actually about what we would expect from our model if
# it was used in a production environment.

# %% [markdown]
# ## Stability of the cross-validation estimates
#
# When doing a single train-test split we don't give any indication regarding
# the robustness of the evaluation of our predictive model: in particular, if
# the test set is small, this estimate of the testing error will be unstable and
# wouldn't reflect the "true error rate" we would have observed with the same
# model on an unlimited amount of test data.
#
# For instance, we could have been lucky when we did our random split of our
# limited dataset and isolated some of the easiest cases to predict in the
# testing set just by chance: the estimation of the testing error would be
# overly optimistic, in this case.
#
# **Cross-validation** allows estimating the robustness of a predictive model by
# repeating the splitting procedure. It will give several training and testing
# errors and thus some **estimate of the variability of the model generalization
# performance**.
#
# There are [different cross-validation
# strategies](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators),
# for now we are going to focus on one called "shuffle-split". At each iteration
# of this strategy we:
#
# - randomly shuffle the order of the samples of a copy of the full dataset;
# - split the shuffled dataset into a train and a test set;
# - train a new model on the train set;
# - evaluate the testing error on the test set.
#
# We repeat this procedure `n_splits` times. Keep in mind that the computational
# cost increases with `n_splits`.
#
# ![Cross-validation diagram](../figures/shufflesplit_diagram.png)
#
# ```{note}
# This figure shows the particular case of **shuffle-split** cross-validation
# strategy using `n_splits=5`.
# For each cross-validation split, the procedure trains a model on all the red
# samples and evaluate the score of the model on the blue samples.
# ```
#
# In this case we will set `n_splits=40`, meaning that we will train 40 models
# in total and all of them will be discarded: we just record their
# generalization performance on each variant of the test set.
#
# To evaluate the generalization performance of our regressor, we can use
# [`sklearn.model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
# with a
# [`sklearn.model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
# object:

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error")

# %% [markdown]
# The results `cv_results` are stored into a Python dictionary. We will convert
# it into a pandas dataframe to ease visualization and manipulation.

# %%
import pandas as pd

cv_results = pd.DataFrame(cv_results)
cv_results.head()

# %% [markdown]
# ```{tip}
# A score is a metric for which higher values mean better results. On the
# contrary, an error is a metric for which lower values mean better results.
# The parameter `scoring` in `cross_validate` always expect a function that is
# a score.
#
# To make it easy, all error metrics in scikit-learn, like
# `mean_absolute_error`, can be transformed into a score to be used in
# `cross_validate`. To do so, you need to pass a string of the error metric
# with an additional `neg_` string at the front to the parameter `scoring`;
# for instance `scoring="neg_mean_absolute_error"`. In this case, the negative
# of the mean absolute error will be computed which would be equivalent to a
# score.
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
# We get timing information to fit and predict at each cross-validation
# iteration. Also, we get the test score, which corresponds to the testing error
# on each of the splits.

# %%
len(cv_results)

# %% [markdown]
# We get 40 entries in our resulting dataframe because we performed 40 splits.
# Therefore, we can show the testing error distribution and thus, have an
# estimate of its variability.

# %%
import matplotlib.pyplot as plt

cv_results["test_error"].plot.hist(bins=10, edgecolor="black", density=True)
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Test error distribution")

# %% [markdown]
# We observe that the testing error is clustered around 47 k\$ and ranges from
# 43 k\$ to 50 k\$.

# %%
print(f"The mean cross-validated testing error is: "
      f"{cv_results['test_error'].mean():.2f} k$")

# %%
print(f"The standard deviation of the testing error is: "
      f"{cv_results['test_error'].std():.2f} k$")

# %% [markdown]
# Note that the standard deviation is much smaller than the mean: we could
# summarize that our cross-validation estimate of the testing error is 46.36 +/-
# 1.17 k\$.
#
# If we were to train a single model on the full dataset (without
# cross-validation) and then later had access to an unlimited amount of test
# data, we would expect its true testing error to fall close to that region.
#
# While this information is interesting in itself, it should be contrasted to
# the scale of the natural variability of the vector `target` in our dataset.
#
# Let us plot the distribution of the target variable:

# %%
target.plot.hist(bins=20, edgecolor="black", density=True)
plt.xlabel("Median House Value (k$)")
_ = plt.title("Target distribution")

# %%
print(f"The standard deviation of the target is: {target.std():.2f} k$")

# %% [markdown]
# The target variable ranges from close to 0 k\$ up to 500 k\$ and, with a
# standard deviation around 115 k\$.
#
# We notice that the mean estimate of the testing error obtained by
# cross-validation is a bit smaller than the natural scale of variation of the
# target variable. Furthermore, the standard deviation of the cross validation
# estimate of the testing error is even smaller.
#
# This is a good start, but not necessarily enough to decide whether the
# generalization performance is good enough to make our prediction useful in
# practice.
#
# We recall that our model makes, on average, an error around 47 k\$. With this
# information and looking at the target distribution, such an error might be
# acceptable when predicting houses with a 500 k\$. However, it would be an
# issue with a house with a value of 50 k\$. Thus, this indicates that our
# metric (Mean Absolute Error) is not ideal.
#
# We might instead choose a metric relative to the target value to predict: the
# mean absolute percentage error would have been a much better choice.
#
# But in all cases, an error of 47 k\$ might be too large to automatically use
# our model to tag house values without expert supervision.
#
# ## More detail regarding `cross_validate`
#
# During cross-validation, many models are trained and evaluated. Indeed, the
# number of elements in each array of the output of `cross_validate` is a
# result from one of these `fit`/`score` procedures. To make it explicit, it is
# possible to retrieve these fitted models for each of the splits/folds by
# passing the option `return_estimator=True` in `cross_validate`.

# %%
cv_results = cross_validate(regressor, data, target, return_estimator=True)
cv_results

# %%
cv_results["estimator"]

# %% [markdown]
# The five decision tree regressors corresponds to the five fitted decision
# trees on the different folds. Having access to these regressors is handy
# because it allows to inspect the internal fitted parameters of these
# regressors.
#
# In the case where you only are interested in the test score, scikit-learn
# provide a `cross_val_score` function. It is identical to calling the
# `cross_validate` function and to select the `test_score` only (as we
# extensively did in the previous notebooks).

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(regressor, data, target)
scores

# %% [markdown]
# ## Summary
#
# In this notebook, we saw:
#
# * the necessity of splitting the data into a train and test set;
# * the meaning of the training and testing errors;
# * the overall cross-validation framework with the possibility to study
#   generalization performance variations.
