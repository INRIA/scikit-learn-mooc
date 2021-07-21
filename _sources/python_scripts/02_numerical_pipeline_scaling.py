# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocessing for numerical features
#
# In this notebook, we will still use only numerical features.
#
# We will introduce these new aspects:
#
# * an example of preprocessing, namely **scaling numerical variables**;
# * using a scikit-learn **pipeline** to chain preprocessing and model
#   training;
# * assessing the generalization performance of our model via **cross-validation**
#   instead of a single train-test split.
#
# ## Data preparation
#
# First, let's load the full adult census dataset.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %%
# to display nice model diagram
from sklearn import set_config
set_config(display='diagram')

# %% [markdown]
# We will now drop the target from the data we will use to train our
# predictive model.

# %%
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

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
# We see that the dataset's features span across different ranges. Some
# algorithms make some assumptions regarding the feature distributions and
# usually normalizing features will be helpful to address these assumptions.
#
# ```{tip}
# Here are some reasons for scaling features:
#
# * Models that rely on the distance between a pair of samples, for instance
#   k-nearest neighbors, should be trained on normalized features to make each
#   feature contribute approximately equally to the distance computations.
#
# * Many models such as logistic regression use a numerical solver (based on
#   gradient descent) to find their optimal parameters. This solver converges
#   faster when the features are scaled.
# ```
#
# Whether or not a machine learning model requires scaling the features depends
# on the model family. Linear models such as logistic regression generally
# benefit from scaling the features while other models such as decision trees
# do not need such preprocessing (but will not suffer from it).
#
# We show how to apply such normalization using a scikit-learn transformer
# called `StandardScaler`. This transformer shifts and scales each feature
# individually so that they all have a 0-mean and a unit standard deviation.
#
# We will investigate different steps used in scikit-learn to achieve such a
# transformation of the data.
#
# First, one needs to call the method `fit` in order to learn the scaling from
# the data.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data_train)

# %% [markdown]
# The `fit` method for transformers is similar to the `fit` method for
# predictors. The main difference is that the former has a single argument (the
# data matrix), whereas the latter has two arguments (the data matrix and the
# target).
#
# ![Transformer fit diagram](../figures/api_diagram-transformer.fit.svg)
#
# In this case, the algorithm needs to compute the mean and standard deviation
# for each feature and store them into some NumPy arrays. Here, these
# statistics are the model states.
#
# ```{note}
# The fact that the model states of this scaler are arrays of means and
# standard deviations is specific to the `StandardScaler`. Other
# scikit-learn transformers will compute different statistics and store them
# as model states, in the same fashion.
# ```
#
# We can inspect the computed means and standard deviations.

# %%
scaler.mean_

# %%
scaler.scale_

# %% [markdown]
# ```{note}
# scikit-learn convention: if an attribute is learned from the data, its name
# ends with an underscore (i.e. `_`), as in `mean_` and `scale_` for the
# `StandardScaler`.
# ```

# %% [markdown]
# Scaling the data is applied to each feature individually (i.e. each column in
# the data matrix). For each feature, we subtract its mean and divide by its
# standard deviation.
#
# Once we have called the `fit` method, we can perform data transformation by
# calling the method `transform`.

# %%
data_train_scaled = scaler.transform(data_train)
data_train_scaled

# %% [markdown]
# Let's illustrate the internal mechanism of the `transform` method and put it
# to perspective with what we already saw with predictors.
#
# ![Transformer transform diagram](../figures/api_diagram-transformer.transform.svg)
#
# The `transform` method for transformers is similar to the `predict` method
# for predictors. It uses a predefined function, called a **transformation
# function**, and uses the model states and the input data. However, instead of
# outputting predictions, the job of the `transform` method is to output a
# transformed version of the input data.

# %% [markdown]
# Finally, the method `fit_transform` is a shorthand method to call
# successively `fit` and then `transform`.
#
# ![Transformer fit_transform diagram](../figures/api_diagram-transformer.fit_transform.svg)

# %%
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled

# %%
data_train_scaled = pd.DataFrame(data_train_scaled,
                                 columns=data_train.columns)
data_train_scaled.describe()

# %% [markdown]
# We can easily combine these sequential operations with a scikit-learn
# `Pipeline`, which chains together operations and is used as any other
# classifier or regressor. The helper function `make_pipeline` will create a
# `Pipeline`: it takes as arguments the successive transformations to perform,
# followed by the classifier or regressor model.

# %%
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
model

# %% [markdown]
# The `make_pipeline` function did not require us to give a name to each step.
# Indeed, it was automatically assigned based on the name of the classes
# provided; a `StandardScaler` will be a step named `"standardscaler"` in the
# resulting pipeline. We can check the name of each steps of our model:

# %%
model.named_steps

# %% [markdown]
# This predictive pipeline exposes the same methods as the final predictor:
# `fit` and `predict` (and additionally `predict_proba`, `decision_function`,
# or `score`).

# %%
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

# %% [markdown]
# We can represent the internal mechanism of a pipeline when calling `fit`
# by the following diagram:
#
# ![pipeline fit diagram](../figures/api_diagram-pipeline.fit.svg)
#
# When calling `model.fit`, the method `fit_transform` from each underlying
# transformer (here a single transformer) in the pipeline will be called to:
#
# - learn their internal model states
# - transform the training data. Finally, the preprocessed data are provided to
#   train the predictor.
#
# To predict the targets given a test set, one uses the `predict` method.

# %%
predicted_target = model.predict(data_test)
predicted_target[:5]

# %% [markdown]
# Let's show the underlying mechanism:
#
# ![pipeline predict diagram](../figures/api_diagram-pipeline.predict.svg)
#
# The method `transform` of each transformer (here a single transformer) is
# called to preprocess the data. Note that there is no need to call the `fit`
# method for these transformers because we are using the internal model states
# computed when calling `model.fit`. The preprocessed data is then provided to
# the predictor that will output the predicted target by calling its method
# `predict`.
#
# As a shorthand, we can check the score of the full predictive pipeline
# calling the method `model.score`. Thus, let's check the computational and
# generalization performance of such a predictive pipeline.

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model[-1].n_iter_[0]} iterations")

# %% [markdown]
# We could compare this predictive model with the predictive model used in
# the previous notebook which did not scale features.

# %%
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
# We see that scaling the data before training the logistic regression was
# beneficial in terms of computational performance. Indeed, the number of
# iterations decreased as well as the training time. The generalization
# performance did not change since both models converged.
#
# ```{warning}
# Working with non-scaled data will potentially force the algorithm to iterate
# more as we showed in the example above. There is also the catastrophic
# scenario where the number of required iterations are more than the maximum
# number of iterations allowed by the predictor (controlled by the `max_iter`)
# parameter. Therefore, before increasing `max_iter`, make sure that the data
# are well scaled.
# ```

# %% [markdown]
# ## Model evaluation using cross-validation
#
# In the previous example, we split the original data into a training set and a 
# testing set. The score of a model will in general depend on the way we make 
# such a split. One downside of doing a single split is that it does not give
# any information about this variability. Another downside, in a setting where 
# the amount of data is small, is that the the data available for training
# and testing will be even smaller after splitting.
#
# Instead, we can use cross-validation. Cross-validation consists of repeating
# the procedure such that the training and testing sets are different each
# time. Generalization performance metrics are collected for each repetition and
# then aggregated. As a result we can get an estimate of the variability of the
# model's generalization performance.
#
# Note that there exists several cross-validation strategies, each of them
# defines how to repeat the `fit`/`score` procedure. In this section, we will
# use the K-fold strategy: the entire dataset is split into `K` partitions. The
# `fit`/`score` procedure is repeated `K` times where at each iteration `K - 1`
# partitions are used to fit the model and `1` partition is used to score. The
# figure below illustrates this K-fold strategy.
#
# ![Cross-validation diagram](../figures/cross_validation_diagram.png)
#
# ```{note}
# This figure shows the particular case of K-fold cross-validation strategy.
# As mentioned earlier, there are a variety of different cross-validation
# strategies. Some of these aspects will be covered in more details in future
# notebooks.
# ```
#
# For each cross-validation split, the procedure trains a model on all the red
# samples and evaluate the score of the model on the blue samples.
# Cross-validation is therefore computationally intensive because it requires
# training several models instead of one.
#
# In scikit-learn, the function `cross_validate` allows to do cross-validation
# and you need to pass it the model, the data, and the target. Since there
# exists several cross-validation strategies, `cross_validate` takes a
# parameter `cv` which defines the splitting strategy.

# %%
# %%time
from sklearn.model_selection import cross_validate

model = make_pipeline(StandardScaler(), LogisticRegression())
cv_result = cross_validate(model, data_numeric, target, cv=5)
cv_result

# %% [markdown]
# The output of `cross_validate` is a Python dictionary, which by default
# contains three entries: (i) the time to train the model on the training data
# for each fold, (ii) the time to predict with the model on the testing data
# for each fold, and (iii) the default score on the testing data for each fold.
#
# Setting `cv=5` created 5 distinct splits to get 5 variations for the training
# and testing sets. Each training set is used to fit one model which is then
# scored on the matching test set. This strategy is called K-fold
# cross-validation where `K` corresponds to the number of splits.
#
# Note that by default the `cross_validate` function discards the 5 models that
# were trained on the different overlapping subset of the dataset. The goal of
# cross-validation is not to train a model, but rather to estimate
# approximately the generalization performance of a model that would have been
# trained to the full training set, along with an estimate of the variability
# (uncertainty on the generalization accuracy).
#
# You can pass additional parameters to `cross_validate` to get more
# information, for instance training scores. These features will be covered in
# a future notebook.
#
# Let's extract the test scores from the `cv_result` dictionary and compute
# the mean accuracy and the variation of the accuracy across folds.

# %%
scores = cv_result["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# Note that by computing the standard-deviation of the cross-validation scores,
# we can estimate the uncertainty of our model generalization performance. This is
# the main advantage of cross-validation and can be crucial in practice, for
# example when comparing different models to figure out whether one is better
# than the other or whether the generalization performance differences are within
# the uncertainty.
#
# In this particular case, only the first 2 decimals seem to be trustworthy. If
# you go up in this notebook, you can check that the performance we get
# with cross-validation is compatible with the one from a single train-test
# split.

# %% [markdown]
# In this notebook we have:
#
# * seen the importance of **scaling numerical variables**;
# * used a **pipeline** to chain scaling and logistic regression training;
# * assessed the generalization performance of our model via **cross-validation**.
