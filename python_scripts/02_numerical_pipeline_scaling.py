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
# datasets, with only numerical features. By contrast, with the previous
# notebook, we will select the numerical features from the original dataset
# instead to read an external file where the selection has been done
# beforehand.
#
# In particular we will highlight:
#
# * an example of preprocessing, namely the **scaling numerical variables**;
# * using a scikit-learn **pipeline** to chain preprocessing and model
#   training;
# * assessing the statistical performance of our model via **cross-validation**
#   instead of a single train-test split.
#
# ## Data preparation
#
# First, let's load the full adult census dataset.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We will now drop the target from the data we will use to train our
# predictive model.

# %%
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

# %% [markdown]
# ```{caution}
# Here and later, we use the name `data` and `target` to be explicit. In
# scikit-learn, documentation `data` is commonly named `X` and `target` is
# commonly called `y`.
# ```

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
#
# ```{caution}
# Be aware that we are using a single train-test split instead of a
# cross-validation to present the scikit-learn transformers API. We are not
# interested in evaluating the statistical performance of the predictive model.
# For this latest purpose, it would be required to evaluate via
# cross-validation.
# ```

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
# algorithms have some assumptions regarding the feature distrbutions and
# usually normalizing features will be helpful to address this assumptions.
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
# First, one needs to call the method `fit`.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data_train)

# %% [markdown]
# The `fit` method is identical to what a predictor is doing.
#
# ![Transformer fit diagram](../figures/api_diagram-transformer.fit.svg)
#
# The scaler uses a learning algorithm. In this case, the algorithm needs to
# compute the mean and standard deviation for each feature and store them into
# some NumPy arrays. Here, these statistics are the model states.
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
# Scaling the data is equivalent to subtract the means and divide by the
# standard deviations previously computed. This operation is defining our
# transformation function and is as well specific to each transformer. We can
# operate this transformation function by calling the method `transform`.

# %%
data_train_scaled = scaler.transform(data_train)
data_train_scaled

# %% [markdown]
# Let's illustrate the internal mechanism of the `transform` method and put it
# to perspective with what we already saw with the predictor.
#
# ![Transformer transform diagram](../figures/api_diagram-transformer.transform.svg)
#
# The `transform` method for the transformer is similar to the `predict` method
# for the predictor. It uses a predefined function, called a **transformation
# function**, and uses the model states and the input data. However, instead of
# outputing predictions, the job of the `transform` method is to output a
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
# followed by the classifier or regressor model, and will assign automatically
# a name at steps based on the name of the classes.

# %%
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# This predictive pipeline exposes the same methods as the ending predictor:
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
# transformer in the pipeline will be called to: (i) learn their internal
# model states and (ii) transform the training data. Finally, the preprocessed
# data are provided to train the predictor.
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
# The method `transform` of each transformer is called to preprocess the data.
# Note that there is no need to call the `fit` method for these transformers
# because we are using the internal model states computed when calling
# `model.fit`. The preprocessed data is then provided to the predictor that
# will output the predicted target by calling its method `predict`.
#
# As a shorthand, we can check the score of the full predictive pipeline
# calling the method `model.score`. Thus, let's check the computational and
# statistical performance of such a predictive pipeline.

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model[-1].n_iter_[0]} iterations")

# %% [markdown]
# We could compare this predictive model with the predictive model used in
# the previous notebook which was not scaling feature.

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
# We see that scaling the data before to train the logistic regression was
# beneficial in terms of computational performance. Indeed, the number of
# iterations decreased as well as the training time. The statistical
# performance did not change since both models converged.
#
# ```{warning}
# Working with non-scaled will potentially force the algorithm to iterate
# more as we showed in the example above. There is also catastrophic scenario
# where the number of iterations required are more than the maximum number of
# iterations allowed by the predictor (controlled by the `max_iter`) parameter.
# Therefore, before to increase `max_iter`, make sure that the data are well
# scaled.
# ```

# %% [markdown]
# ## Model evaluation using cross-validation
#
# In the previous example, we split the original data into a training set and a
# testing set. This strategy has several issues: in the setting where the
# amount of data is limited, the subset used to train or test will be small.
# Moreover, if the splitting was done in a random manner, we do not have
# information regarding the confidence of the results obtained.
#
# Instead, we can use cross-validation. Cross-validation consists of repeating
# this random splitting into training and testing sets and aggregating the
# model statistical performance. By repeating the experiment, one can get an
# estimate of the variability of the model statistical performance.
#
# ```{note}
# We will go into details regarding cross-validation in the upcoming module
# "Selecting the best model".
# ```
#
# The function `cross_validate` allows for such experimental protocol by
# providing the model, the data, and the target. Since there exists several
# cross-validation strategies, `cross_validate` takes a parameter `cv` which
# defines the splitting strategy.

# %%
# %%time
from sklearn.model_selection import cross_validate

model = make_pipeline(StandardScaler(), LogisticRegression())
cv_result = cross_validate(model, data_numeric, target, cv=5)
cv_result

# %% [markdown]
# The output of `cross_validate` contains by default three entries: (i) the
# time to train the model on the training data for each fold, (ii) the time
# to predict with the model on the testing data for each fold, and (iii) the
# default score on the testing data for each fold.
#
# Additional can be returned, for instance training scores or the fitted models
# per fold, by passing additional parameters. We will give more details about
# these features in a subsequent notebook.
#
# Let's extract the test scores for the dictionary and compute the mean
# accuracy and the variation of the accuracy across folds.

# %%
scores = cv_result["test_score"]
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
# Note that by default the `cross_validate` function above discards the 5
# models that were trained on the different overlapping subset of the dataset.
# The goal of cross-validation is not to train a model, but rather to estimate
# approximately the generalization performance of a model that would have been
# trained to the full training set, along with an estimate of the variability
# (uncertainty on the generalization accuracy).

# %% [markdown]
# In this notebook we have:
#
# * seen the importance of **scaling numerical variables**;
# * used a **pipeline** to chain scaling and logistic regression training;
# * assessed the statistical performance of our model via **cross-validation**.
