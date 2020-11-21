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
# We will use this subset of data to fit a linear classification model to
# predict the income class.

# %%
data.columns

# %% [markdown]
# When building a machine learning model, it is important to leave out a
# subset of the data which we can use later to evaluate the trained model.
# The data used to fit a model is called training data while the one used to
# assess a model is called testing data.
#
# Scikit-learn provides a helper function `train_test_split` which will
# split the dataset into a training and a testing set. It will also ensure that
# the data are shuffled randomly before splitting the data.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %%
print(
    f"The training dataset contains {data_train.shape[0]} samples and "
    f"{data_train.shape[1]} features")

# %%
print(
    f"The testing dataset contains {data_test.shape[0]} samples and "
    f"{data_test.shape[1]} features")

# %% [markdown]
# We will build a linear classification model called "Logistic Regression". The
# `fit` method is called to train the model from the input (features) and
# target data. Only the training data should be given for this purpose.
#
# In addition, check the time required to train the model and the number of
# iterations done by the solver to find a solution.
# %%
from sklearn.linear_model import LogisticRegression
import time

model = LogisticRegression()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

print(f"The model {model.__class__.__name__} was trained in "
      f"{elapsed_time:.3f} seconds for {model.n_iter_} iterations")

# %% [markdown]
# Let's ignore the convergence warning for now and instead let's try
# to use our model to make some predictions on the first five records
# of the held out test set:

# %%
target_predicted = model.predict(data_test)
target_predicted[:5]

# %%
target_test[:5]

# %%
predictions = data_test.copy()
predictions['predicted-class'] = target_predicted
predictions['expected-class'] = target_test
predictions['correct'] = target_predicted == target_test
predictions.head()

# %% [markdown]
# To quantitatively evaluate our model, we can use the method `score`. It will
# compute the classification accuracy when dealing with a classification
# problem.

# %%
print(f"The test accuracy using a {model.__class__.__name__} is "
      f"{model.score(data_test, target_test):.3f}")

# %% [markdown]
# This is mathematically equivalent as computing the average number of time
# the model makes a correct prediction on the test set:

# %%
(target_test == target_predicted).mean()

# %% [markdown]
# In this notebook we have:
# * **split** our dataset into a training dataset and a testing dataset to eva
# * fitted a **logistic regression** model on the training data
# * evaluated its performance on the testing data
# * presented the scikit-learn API `.fit` (to train a model), `.predict` (to
#   make predictions) and `.score` (to evaluate a model)
