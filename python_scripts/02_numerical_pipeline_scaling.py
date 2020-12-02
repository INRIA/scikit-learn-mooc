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
#
# In this notebook, we present: how to build predictive models on tabular
# datasets, with only numerical features.
#
# In particular we will highlight:
#
# * an example of preprocessing, namely the **scaling numerical variables**;
# * using a scikit-learn **pipeline** to chain preprocessing and model
#   training;
# * assessed the performance of our model via **cross-validation**.
#
# # Preprocessing for numerical features

# %% [markdown]
# Let's charge the full adult census dataset and split the target from the
# actual data used for modeling.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = df[target_name]

data = df.drop(columns=[target_name, "fnlwgt"])

# %% [markdown]
# In this notebook, we will only focus on the numerical processing. Thus, we
# need to recognize and select the columns corresponding to numerical features.
# We can first investigate the type of data for each of the columns.

# %%
data.dtypes

# %%
data.dtypes.unique()

# %% [markdown]
# Thus, we see two types of data types: integer and object. We can look at
# the first few lines of the dataframe to understand the meaning of the
# `object` data type.

# %%
data.head()

# %% [markdown]
# We can see that `object` data type corresponds to columns containing string.
# As we saw in the exploration section, these columns contains categories and
# we will see later how to handle those. We can select the columns containing
# integers and check their content.

# %%
numerical_columns = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
data[numerical_columns].head()

# %% [markdown]
# Now that we limited the dataset to numerical columns only, we should check
# that none of the columns are containing categorical columns using integer.
# Indeed, the column `"education-num"` represents such information. Thus, we
# need to exclude it from the columns to consider.

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
# Let's consider the `ConvergenceWarning` message that was raised previously
# when calling the `fit` method to train our model. This warning informs us
# that our model stopped learning because it reached the maximum number of
# iterations allowed by the user. This could potentially be detrimental for the
# model accuracy. We can follow the (bad) advice given in the warning message
# and increase the maximum number of iterations allowed.

# %%
from sklearn.linear_model import LogisticRegression
import time

model = LogisticRegression(max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model.n_iter_} iterations")

# %% [markdown]
# We now observe a longer training time but no significant improvement in the
# predictive performance. Instead of increasing the number of iterations, we
# can try to help fit the model faster by scaling the data first. A range of
# preprocessing algorithms in scikit-learn allows us to transform the input
# data before training a model.
#
# In our case, we will standardize the data and then train a new logistic
# regression model on that new version of the dataset.

# %%
data_train.describe()

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
# `Pipeline` by giving as arguments the successive transformations to perform
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
# shorter while the predictive performance (accuracy) slightly improve.
#
# In the previous example, we split the original data into a training set and a
# testing set. This strategy has several issues: in the setting where the
# amount of data is limited, the subset of data used to train or test will be
# small; and the splitting was done in a random manner and we have no
# information regarding the confidence of the results obtained.
#
# Instead, we can use cross-validation. Cross-validation consists of repeating
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

# %%
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
