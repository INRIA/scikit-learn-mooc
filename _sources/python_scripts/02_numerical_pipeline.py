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
# ## Basic preprocessing and model fitting
#
# In this notebook, we present how to build predictive models on tabular
# datasets, with only numerical features.
#
# In particular we will highlight:
#
# * the scikit-learn API : `.fit(X, y)`/`.predict(X)`/`.score(X, y)`;
# * how to evaluate the performance of a model with a train-test split.
#
# ## Loading the dataset
#
# We will use the same dataset "adult_census" described in the previous
# notebook. For more details about the dataset see
# <http://www.openml.org/d/1590>.
#
# Numerical data is the most natural type of data used in machine learning and
# can (almost) directly be fed into predictive models. We will load a the
# subset of the original data with only the numerical columns.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census-numeric.csv")

# %% [markdown]
# Let's have a look at the first records of this data frame:

# %%
df.head()

# %% [markdown]
# We see that this CSV file contains all information: the target that we would
# like to predict (i.e. `"class"`) and the data that we want to use to train
# our predictive model (i.e. the remaining columns). The first step is to
# split our entire dataset to get on one side the target and on the other side
# the data.

# %%
target_name = "class"
target = df[target_name]
target

# %%
data = df.drop(columns=[target_name, ])
data.head()

# %% [markdown]
# We can now linger on the variables, also denominated features, that we will
# use to build our predictive model. In addition, we can as well check how many
# samples are available in our dataset.

# %%
data.columns

# %%
print(
    f"The dataset contains {data.shape[0]} samples and "
    f"{data.shape[1]} features")

# %% [markdown]
# We will build a classification model using the "K-nearest neighbors"
# strategy. The `fit` method is called to train the model from the input
# (features) and target data.

# %%
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(data, target)

# %% [markdown]
# Let's use our model to make some predictions using the same dataset. In a
# sake of simplicity, we will look at the five first predicted targets.

# %%
target_predicted = model.predict(data)
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
# Here, we see that our model does a mistake when predicting for the first
# sample.
#
# To get a better assessment, we can compute the average success rate.

# %%
(target == target_predicted).mean()

# %% [markdown]
# But, can this evaluation be trusted, or is it too good to be true?
#
# When building a machine learning model, it is important evaluate the trained
# model on data that was not used to fit the model, as generalization is more
# than memorization. It is harder to conclude on instances never seen than on
# those already seen.
#
# Correct evaluation is easily done by leaving out a subset of the data when
# training the model and using it after for model evaluation. The data used to
# fit a model is called training data while the one used to assess a model is
# called testing data.
#
# We can load more data, which was actually left-out from the original data
# set.

# %%
df_test = pd.read_csv('../datasets/adult-census-numeric-test.csv')

# %% [markdown]
# From this new data, we separate out input features and the target to predict,
# as in the beginning of this notebook.

# %%
target_test = df_test[target_name]
data_test = df_test.drop(columns=[target_name, ])

# %% [markdown]
# We can check the number of features and samples available in this new set.

# %%
print(
    f"The testing dataset contains {data_test.shape[0]} samples and "
    f"{data_test.shape[1]} features")

# %% [markdown]
# Note that scikit-learn provides a helper function `train_test_split` which
# can be used to split the dataset into a training and a testing set. It will
# also ensure that the data are shuffled randomly before splitting the data.


# %% [markdown]
# Instead of computing the prediction and computing manually the average
# success rate, we can use the method `score`. When dealing with classifiers
# this method return this performance metric.

# %%
accuracy = model.score(data_test, target_test)
model_name = model.__class__.__name__

print(f"The test accuracy using a {model_name} is "
      f"{accuracy:.3f}")

# %% [markdown]
# If we compare with the accuracy obtained by wrongly evaluating the model
# on the training set, we find that this evaluation was indeed optimistic
# compared to the score obtained on an held-out test set.
#
# It shows the importance to always test the performance of predictive models
# on a different set than the one used to train these models. We will come
# back more into details regarding how predictive models should be evaluated.

# %% [markdown]
# In this notebook we have:
#
# * fit a **k-nearest neighbors** model on training dataset;
# * evaluated its performance on the testing data;
# * presented the scikit-learn API `.fit(X, y)` (to train a model),
#   `.predict(X)` (to make predictions) and `.score(X, y)`
#   (to evaluate a model).
