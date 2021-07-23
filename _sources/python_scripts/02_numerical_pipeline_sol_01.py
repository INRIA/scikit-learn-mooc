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
# # 📃 Solution for Exercise M1.03
#
# The goal of this exercise is to compare the performance of our classifier in
# the previous notebook (roughly 81% accuracy with `LogisticRegression`) to
# some simple baseline classifiers. The simplest baseline classifier is one
# that always predicts the same class, irrespective of the input data.
#
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <=50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
#
# Use a `DummyClassifier` and do a train-test split to evaluate
# its accuracy on the test set. This
# [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
# shows a few examples of how to evaluate the generalization performance of these
# baseline models.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We will first split our dataset to have the target separated from the data
# used to train our predictive model.

# %%
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

# %% [markdown]
# We start by selecting only the numerical columns as seen in the previous
# notebook.

# %%
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

# %% [markdown]
# Split the data and target into a train and test set.

# %%
from sklearn.model_selection import train_test_split

# solution
data_numeric_train, data_numeric_test, target_train, target_test = \
    train_test_split(data_numeric, target, random_state=42)

# %% [markdown]
# Use a `DummyClassifier` such that the resulting classifier will always
# predict the class `' >50K'`. What is the accuracy score on the test set?
# Repeat the experiment by always predicting the class `' <=50K'`.
#
# Hint: you can set the `strategy` parameter of the `DummyClassifier` to
# achieve the desired behavior.

# %%
from sklearn.dummy import DummyClassifier

# solution
class_to_predict = " >50K"
high_revenue_clf = DummyClassifier(strategy="constant",
                                   constant=class_to_predict)
high_revenue_clf.fit(data_numeric_train, target_train)
score = high_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting only high revenue: {score:.3f}")

# %% [markdown] tags=["solution"]
# We clearly see that the score is below 0.5 which might be surprising at
# first. We will now check the generalization performance of a model which always
# predict the low revenue class, i.e. `" <=50K"`.

# %% tags=["solution"]
class_to_predict = " <=50K"
low_revenue_clf = DummyClassifier(strategy="constant",
                                  constant=class_to_predict)
low_revenue_clf.fit(data_numeric_train, target_train)
score = low_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting only low revenue: {score:.3f}")

# %% [markdown] tags=["solution"]
# We observe that this model has an accuracy higher than 0.5. This is due to
# the fact that we have 3/4 of the target belonging to low-revenue class.

# %% [markdown] tags=["solution"]
# Therefore, any predictive model giving results below this dummy classifier
# will not be helpful.

# %% tags=["solution"]
adult_census["class"].value_counts()

# %% tags=["solution"]
(target == " <=50K").mean()

# %% [markdown] tags=["solution"]
# In practice, we could have the strategy `"most_frequent"` to predict the
# class that appears the most in the training target.

# %% tags=["solution"]
most_freq_revenue_clf = DummyClassifier(strategy="most_frequent")
most_freq_revenue_clf.fit(data_numeric_train, target_train)
score = most_freq_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting the most frequent class: {score:.3f}")

# %% [markdown] tags=["solution"]
# So the `LogisticRegression` accuracy (roughly 81%) seems better than the
# `DummyClassifier` accuracy (roughly 76%). In a way it is a bit reassuring,
# using a machine learning model gives you a better performance than always
# predicting the majority class, i.e. the low income class `" <=50K"`.
