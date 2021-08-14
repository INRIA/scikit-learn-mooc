# ---
# jupyter:
#   jupytext:
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
# # First model with scikit-learn
#
# In this notebook, we present how to build predictive models on tabular
# datasets, with only numerical features.
#
# In particular we will highlight:
#
# * the scikit-learn API: `.fit(X, y)`/`.predict(X)`/`.score(X, y)`;
# * how to evaluate the generalization performance of a model with a train-test
#   split.
#
# ## Loading the dataset with Pandas
#
# We will use the same dataset "adult_census" described in the previous
# notebook. For more details about the dataset see
# <http://www.openml.org/d/1590>.
#
# Numerical data is the most natural type of data used in machine learning and
# can (almost) directly be fed into predictive models. We will load a
# subset of the original data with only the numerical columns.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")

# %% [markdown]
# Let's have a look at the first records of this dataframe:

# %%
adult_census.head()

# %% [markdown]
# We see that this CSV file contains all information: the target that we would
# like to predict (i.e. `"class"`) and the data that we want to use to train
# our predictive model (i.e. the remaining columns). The first step is to
# separate columns to get on one side the target and on the other side the
# data.
#
# ## Separate the data and the target

# %%
target_name = "class"
target = adult_census[target_name]
target

# %%
data = adult_census.drop(columns=[target_name, ])
data.head()

# %% [markdown]
# We can now linger on the variables, also denominated features, that we will
# use to build our predictive model. In addition, we can also check how many
# samples are available in our dataset.

# %%
data.columns

# %%
print(f"The dataset contains {data.shape[0]} samples and "
      f"{data.shape[1]} features")

# %% [markdown]
# ## Fit a model and make predictions
#
# We will build a classification model using the "K-nearest neighbors"
# strategy. To predict the target of a new sample, a k-nearest neighbors takes
# into account its `k` closest samples in the training set and predicts the
# majority target of these samples.
#
# ```{caution}
# We use a K-nearest neighbors here. However, be aware that it is seldom useful
# in practice. We use it because it is an intuitive algorithm. In the next
# notebook, we will introduce better models.
# ```
#
# The `fit` method is called to train the model from the input (features) and
# target data.

# %%
# to display nice model diagram
from sklearn import set_config
set_config(display='diagram')

# %%
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(data, target)

# %% [markdown]
# Learning can be represented as follows:
#
# ![Predictor fit diagram](../figures/api_diagram-predictor.fit.svg)
#
# The method `fit` is composed of two elements: (i) a **learning algorithm**
# and (ii) some **model states**. The learning algorithm takes the training
# data and training target as input and sets the model states. These model
# states will be used later to either predict (for classifiers and regressors)
# or transform data (for transformers).
#
# Both the learning algorithm and the type of model states are specific to each
# type of model.

# %% [markdown]
# ```{note}
# Here and later, we use the name `data` and `target` to be explicit. In
# scikit-learn documentation, `data` is commonly named `X` and `target` is
# commonly called `y`.
# ```

# %% [markdown]
# Let's use our model to make some predictions using the same dataset.

# %%
target_predicted = model.predict(data)

# %% [markdown]
# We can illustrate the prediction mechanism as follows:
#
# ![Predictor predict diagram](../figures/api_diagram-predictor.predict.svg)
#
# To predict, a model uses a **prediction function** that will use the input
# data together with the model states. As for the learning algorithm and the
# model states, the prediction function is specific for each type of model.

# %% [markdown]
# Let's now have a look at the computed predictions. For the sake of
# simplicity, we will look at the five first predicted targets.

# %%
target_predicted[:5]

# %% [markdown]
# Indeed, we can compare these predictions to the actual data...

# %%
target[:5]

# %% [markdown]
# ...and we could even check if the predictions agree with the real targets:

# %%
target[:5] == target_predicted[:5]

# %%
print(f"Number of correct prediction: "
      f"{(target[:5] == target_predicted[:5]).sum()} / 5")

# %% [markdown]
# Here, we see that our model makes a mistake when predicting for the first
# sample.
#
# To get a better assessment, we can compute the average success rate.

# %%
(target == target_predicted).mean()

# %% [markdown]
# But, can this evaluation be trusted, or is it too good to be true?
#
# ## Train-test data split
#
# When building a machine learning model, it is important to evaluate the
# trained model on data that was not used to fit it, as **generalization** is
# more than memorization (meaning we want a rule that generalizes to new data,
# without comparing to data we memorized).
# It is harder to conclude on never-seen instances than on already seen ones.
#
# Correct evaluation is easily done by leaving out a subset of the data when
# training the model and using it afterwards for model evaluation.
# The data used to fit a model is called training data while the data used to
# assess a model is called testing data.
#
# We can load more data, which was actually left-out from the original data
# set.

# %%
adult_census_test = pd.read_csv('../datasets/adult-census-numeric-test.csv')

# %% [markdown]
# From this new data, we separate our input features and the target to predict,
# as in the beginning of this notebook.

# %%
target_test = adult_census_test[target_name]
data_test = adult_census_test.drop(columns=[target_name, ])

# %% [markdown]
# We can check the number of features and samples available in this new set.

# %%
print(f"The testing dataset contains {data_test.shape[0]} samples and "
      f"{data_test.shape[1]} features")

# %% [markdown]
#
# Instead of computing the prediction and manually computing the average
# success rate, we can use the method `score`. When dealing with classifiers
# this method returns their performance metric.

# %%
accuracy = model.score(data_test, target_test)
model_name = model.__class__.__name__

print(f"The test accuracy using a {model_name} is "
      f"{accuracy:.3f}")

# %% [markdown]
# Let's check the underlying mechanism when the `score` method is called:
#
# ![Predictor score diagram](../figures/api_diagram-predictor.score.svg)
#
# To compute the score, the predictor first computes the predictions (using
# the `predict` method) and then uses a scoring function to compare the
# true target `y` and the predictions. Finally, the score is returned.

# %% [markdown]
# If we compare with the accuracy obtained by wrongly evaluating the model
# on the training set, we find that this evaluation was indeed optimistic
# compared to the score obtained on a held-out test set.
#
# It shows the importance to always testing the generalization performance of
# predictive models on a different set than the one used to train these models.
# We will discuss later in more details how predictive models should be
# evaluated.

# %% [markdown]
# ```{note}
# In this MOOC, we will refer to **generalization performance** of a model when
# referring to the test score or test error obtained by comparing the
# prediction of a model and the true targets. Equivalent terms for
# **generalization performance** are predictive performance and statistical
# performance. We will refer to **computational performance** of a predictive
# model when accessing the computational costs of training a predictive model
# or using it to make predictions.
# ```

# %% [markdown]
# In this notebook we:
#
# * fitted a **k-nearest neighbors** model on a training dataset;
# * evaluated its generalization performance on the testing data;
# * introduced the scikit-learn API `.fit(X, y)` (to train a model),
#   `.predict(X)` (to make predictions) and `.score(X, y)`
#   (to evaluate a model).
