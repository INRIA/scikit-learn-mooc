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
# * the scikit-learn API : `.fit`/`.predict`/`.score`
# * how to evaluate the performance of a model with a train-test split
#
# ## Loading the dataset
#
# We will use the same dataset "adult_census" described in the previous notebook.
# For more details about the dataset see <http://www.openml.org/d/1590>.
#
# Numerical data is the most natural type of data used in machine
# learning and can (almost) directly be fed into predictive models. We
# will load a the subset of the original data with only the numerical
# columns.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census-numeric.csv")

# %% [markdown]
# Let's have a look at the first records of this data frame:

# %%
df.head()

# %%
target_name = "class"
target = df[target_name]
target

# %% [markdown]
# We now separate out the data that we will use to predict from the
# prediction target

# %%
data = df.drop(columns=[target_name, ])
data.head()


# %% [markdown]
# We will use this data to fit a linear classification model to predict
# the income class.

# %%
data.columns

# %%
print(
    f"The dataset contains {data.shape[0]} samples and "
    f"{data.shape[1]} features")

# %% [markdown]
# We will build a classification model using the "K Nearest Neighbor"
# strategy. The `fit` method is called to train the model from the input
# (features) and target data.
# %%
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(data, target)

# %% [markdown]
# Let'us to use our model to make some predictions on the first five
# records of the held out test set:

# %%
target_predicted = model.predict(data)
target_predicted[:5]

# %% [markdown]
# We can compare these predictions to the actual data

# %%
utarget[:5]

# %% [markdown]
# To get a better assessment, we can compute the average success rate

# %%
(target == target_predicted).mean()

# %% [markdown]
# But, can this evaluation be trusted, or is it too good to be true?
#
# When building a machine learning model, it is important evaluate the
# trained model on data that was not used to fit the model, as
# generalization is more than memorization. It is harder to conclude on
# instances never seen than on those already seen.
#
# Correct evaluation is easily done by leaving out a subset of the data
# when training the model and using it after for model evaluation. The
# data used to fit a model is called training data while the one used to
# assess a model is called testing data.
#
# We can load more data, which was actually left-out from the original
# data set

# %%
df_test = pd.read_csv('../datasets/adult-census-numeric-test.csv')

# %% [markdown]
# From this new data, we separate out input features and the target to
# predict

# %%
target_test = df_test[target_name]
data_test = df_test.drop(columns=[target_name, ])

# %%
print(
    f"The testing dataset contains {data_test.shape[0]} samples and "
    f"{data_test.shape[1]} features")

# %% [markdown]
# Note that scikit-learn provides a helper function `train_test_split`
# which can be used to split the dataset into a training and a testing
# set. It will also ensure that the data are shuffled randomly before
# splitting the data.


# %% [markdown]
# To quantitatively evaluate our model, we can use the method `score`. It will
# compute the classification accuracy when dealing with a classification
# problem.

# %%
print(f"The test accuracy using a {model.__class__.__name__} is "
      f"{model.score(data_test, target_test):.3f}")

# %% [markdown]
# We can now compute the model predictions on the test set:

# %%
target_test_predicted = model.predict(data_test)

# %% [markdown]
# And compute the average accuracy on the test set:

# %%
(target_test == target_test_predicted).mean()

# %% [markdown]
# If we compare with the accuracy obtained by wrongly evaluating the model
# on the training set, we find that this evaluation was indeed optimistic

# %% [markdown]
# In this notebook we have:
# * **split** our dataset into a training dataset and a testing dataset to eva
# * fitted a **nearest neighbor** model on the training data
# * evaluated its performance on the testing data
# * presented the scikit-learn API `.fit` (to train a model), `.predict` (to
#   make predictions) and `.score` (to evaluate a model)
