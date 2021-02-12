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
# # Preprocessing for numerical features
#
# In this notebook, we present how to build predictive models on tabular
# datasets, with only numerical features.
#
# In particular we will highlight:
#
# * an example of preprocessing, namely the **scaling numerical variables**;
# * using a scikit-learn **pipeline** to chain preprocessing and model
#   training;
# * assessing the performance of our model via **cross-validation** instead of
#   a single train-test split.
#
# ## Data preparation
#
# First, let's charge the full adult census dataset.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We will now drop the target from the data we will use to train our
# predictive model.

# %%
target_name = "class"
target = df[target_name]
data = df.drop(columns=target_name)

# %% [markdown]
# Then, we select only the numerical columns, as seen in the previous
# notebook.

# %%
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

# %% [markdown]
# Finally, we can divide our dataset into a train and test sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42)

# %% [markdown]
# ## Model fitting without preprocessing
#
# We will use the logistic regression classifier as in the previous notebook.
# This time, besides fitting the model, we will also compute the time needed
# to train the model.

# %%
import time
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model.n_iter_[0]} iterations")

# %% [markdown]
# We did not have issues with our model training: it converged in 59 iterations
# while we gave maximum number of iterations of 100. However, we can give an
# hint that this model could converge and train faster. In some case, we might
# even get a `ConvergenceWarning` using the above pattern. We will show such
# example by reducing the number of maximum iterations allowed.

# %%
model = LogisticRegression(max_iter=50)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model.n_iter_[0]} iterations")

# %% [markdown]
# In this case, the score is closed to the previous case because our algorithm
# is closed to the same solution. The warning suggested by scikit-learn
# provides two solutions to solve this issue:
#
# * increase `max_iter` which is indeed what happens in the former case. We let
#   the algorithm converge, it will take more time but we are sure to get an
#   optimal model;
# * standardize the data which is expected to improve convergence.
#
# We will investigate the second option.
#
# ## Model fitting with preprocessing
#
# A range of preprocessing algorithms in scikit-learn allow us to transform
# the input data before training a model. In our case, we will standardize the
# data and then train a new logistic regression model on that new version of
# the dataset.
#
# Let's start by printing some statistics about the training data.

# %%
data_train.describe()

# %% [markdown]
# We see that the dataset's features span across different ranges.
# However, we want to have normalized features because some algorithms make
# assumptions regarding their distribution, and being normalized is usually
# one of them.
#
# ```{tip}
# Here are some reasons for scaling features:
#
# * Models that rely on the distance between a pair of samples (e.g k-nearest
#   neighbors) should be trained on normalized features to make each feature
#   contribute approximately equally to the distance computations.
#
# * Many models such as logistic regression use a numerical solver (based on
#   gradient descent) to find their optimal parameters. This solver converges
#   faster when the features are scaled.
#
# * predictors using Euclidean distance (e.g k-nearest-neighbors) should have
#   normalized features so that each one contributes equally to the distance
#   computation;
# * predictors using gradient-descent based algorithms (e.g logistic regression)
#   to find optimal parameters work better and faster;
# * predictors using regularization (e.g logistic regression) require normalized
#   features to properly apply the weights.
# ```
#
# Whether or not a machine learning model requires scaling the features depends
# on the model family. Linear models such as logistic regression generally
# benefit from scaling the features while other models such as decision trees do
# not need such preprocessing.
#
# We show how to apply such normalization using a scikit-learn transformer
# called `StandardScaler`. This transformer shifts and scales each feature
# individually so that they all have a 0-mean and a unit standard deviation.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled

# %%
data_train_scaled = pd.DataFrame(data_train_scaled,
                                 columns=data_train.columns)
data_train_scaled.describe()

# %% [markdown]
# We can easily combine these sequential operations with a scikit-learn
# `Pipeline`, which chains together operations and can be used like any other
# classifier or regressor. The helper function `make_pipeline` will create a
# `Pipeline`: it takes as arguments the successive transformations to perform,
# followed by the classifier or regressor model.

# %%
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model[-1].n_iter_} iterations")

# %% [markdown]
# We can see that the training time and the number of iterations is much
# shorter while the predictive performance (accuracy) slightly improved.
#
# ## Model evaluation using cross-validation
#
# In the previous example, we split the original data into a training set
# and a testing set.
# This strategy has several issues: in the setting where the amount of data
# is limited, the subset used to train or test will be small.
# Moreover, if the splitting was done in a random manner, we do not have
# information regarding the confidence of the results obtained.
#
# Instead, we can use cross-validation.
# Cross-validation consists of repeating
# this random splitting into training and testing sets and aggregating the
# model performance. By repeating the experiment, one can get an estimate of
# the variability of the model performance.
#
# The function `cross_val_score` allows for such experimental protocol by
# giving the model, the data and the target. Since there exists several
# cross-validation strategies, `cross_val_score` takes a parameter `cv` which
# defines the splitting strategy.

# %%
# %%time
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data_numeric, target, cv=5)
scores

# %%
print(f"The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# Note that by computing the standard-deviation of the cross-validation scores
# we can get an idea of the uncertainty of our estimation of the predictive
# performance of the model: in the above results, only the first 2 decimals
# seem to be trustworthy. Using a single train / test split would not allow us
# to know anything about the level of uncertainty of the accuracy of the model.
#
# Setting `cv=5` created 5 distinct splits to get 5 variations for the training
# and testing sets. Each training set is used to fit one model which is then
# scored on the matching test set. This strategy is called K-fold
# cross-validation where `K` corresponds to the number of splits.
#
# The figure helps visualize how the dataset is partitioned into train and test
# samples at each iteration of the cross-validation procedure:
#
# ![Cross-validation diagram](../figures/cross_validation_diagram.png)
#
# For each cross-validation split, the procedure trains a model on the
# concatenation of the red samples and evaluate the score of the model by using
# the blue samples. Cross-validation is therefore computationally intensive
# because it requires training several models instead of one.
#
# Note that the `cross_val_score` method above discards the 5 models that were
# trained on the different overlapping subset of the dataset. The goal of
# cross-validation is not to train a model, but rather to estimate
# approximately the generalization performance of a model that would have been
# trained to the full training set, along with an estimate of the variability
# (uncertainty on the generalization accuracy).

# %% [markdown]
# In this notebook we have:
#
# * seen the importance of **scaling numerical variables**;
# * used a **pipeline** to chain scaling and logistic regression training;
# * assessed the performance of our model via **cross-validation**.
