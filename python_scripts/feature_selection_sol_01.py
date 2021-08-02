# %% [markdown]
# # 📃 Solution for Exercise 01
#
# The aim of this exercise is to highlight caveats to have in mind when using
# feature selection. You have to be extremely careful regarding the set of
# data on which you will compute the statistic that help you feature algorithm
# to decide which feature to select.
#
# On purpose, we will make you program the wrong way of doing feature selection
# to insights.
#
# First, you will create a completely random dataset using NumPy. Using the
# function `np.random.randn`, generate a matrix `data` containing 100 samples
# and 100,000 features. Then, using the function `np.random.randint`, generate
# a vector `target` with 100 samples containing either 0 or 1.
#
# This type of dimensionality is typical in bioinformatics when dealing with
# RNA-seq. However, we will use completely randomized features such that we
# don't have a link between the data and the target. Thus, the generalization
# performance of any machine-learning model should not perform better than the
# chance-level.

# %%
import numpy as np

# solution
rng = np.random.RandomState(42)
data, target = rng.randn(100, 100000), rng.randint(0, 2, size=100)

# %% [markdown]
# Now, create a logistic regression model and use cross-validation to check
# the score of such model. It will allow use to confirm that our model cannot
# predict anything meaningful from random data.

# %%
# solution
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# solution
model = LogisticRegression()
test_score = cross_val_score(model, data, target, n_jobs=2)
print(f"The mean accuracy is: {test_score.mean():.3f}")

# %% [markdown] tags=["solution"]
# It is not surprising that the logistic regression model performs as bad as
# pure chance when we provide the full dataset.

# %% [markdown]
# Now, we will ask you to program the **wrong** pattern to select feature.
# Select the feature by using the entire dataset. We will choose ten features
# with the highest ANOVA F-score computed on the full dataset. Subsequently,
# subsample the dataset `data` by selecting the features' subset. Finally,
# train and test a logistic regression model.
#
# You should get some surprising results.

# %%
# solution
from sklearn.feature_selection import SelectKBest, f_classif

# solution
feature_selector = SelectKBest(score_func=f_classif, k=10)
data_subset = feature_selector.fit_transform(data, target)
test_score = cross_val_score(model, data_subset, target)
print(f"The mean accuracy is: {test_score.mean():.3f}")

# %% [markdown] tags=["solution"]
# Surprisingly, the logistic regression succeeded in having a fantastic
# accuracy using data with no link with the target, initially. We, therefore,
# know that these results are not legit.
#
# The reasons for obtaining these results are two folds: the pool of available
# features is large compared to the number of samples. It is possible to find a
# subset of features that will link the data and the target. By not splitting
# the data, we leak knowledge from the entire dataset and could use this
# knowledge will evaluating our model.

# %% [markdown]
# Now, we will make you program the **right** way to do the feature selection.
# First, split the dataset into a training and testing set. Then, fit the
# feature selector on the training set. Then, transform both the training and
# testing sets before to train and test the logistic regression.

# %%
# solution
from sklearn.model_selection import train_test_split

# solution
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)
feature_selector.fit(data_train, target_train)
data_train_subset = feature_selector.transform(data_train)
data_test_subset = feature_selector.transform(data_test)
model.fit(data_train_subset, target_train)
test_score = model.score(data_test_subset, target_test)
print(f"The mean accuracy is: {test_score:.3f}")

# %% [markdown] tags=["solution"]
# It is not a surprise that our model is not working. We see that selecting
# features only on the training set will not help when testing our model. In
# this case, we obtained the expected results.
#
# Therefore, as with hyperparameters optimization or model selection, tuning
# the feature space should be done solely on the training set, keeping a part
# of the data left-out.
#

# %% [markdown]
# However, the previous case is not perfect. For instance, if we were asking
# to perform cross-validation, the manual `fit`/`transform` of the datasets
# will make our life hard. Indeed, the solution here is to use a scikit-learn
# pipeline in which the feature selection will be a pre processing stage
# before to train the model.
#
# Thus, start by creating a pipeline with the feature selector and the logistic
# regression. Then, use cross-validation to get an estimate of the uncertainty
# of your model generalization performance.

# %%
# solution
from sklearn.pipeline import make_pipeline

# solution
model = make_pipeline(feature_selector, LogisticRegression())
test_score = cross_val_score(model, data, target)
print(f"The mean accuracy is: {test_score.mean():.3f}")

# %% [markdown] tags=["solution"]
# We see that using a scikit-learn pipeline is removing a lot of boilerplate
# code and avoiding to make mistake while calling `fit` and `transform` on the
# different set of data.
