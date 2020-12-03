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
# # 📃 Solution for Exercise 01
#
# The goal of this exercise is to compare the performance of our classifier
# (81% accuracy) to some baseline classifiers that would ignore the input data
# and instead make constant predictions.
#
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <= 50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
#
# Use a `DummyClassifier` and do a train-test split to evaluate
# its accuracy on the test set. This
# [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
# shows a few examples of how to evaluate the performance of these baseline
# models.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We will first split our dataset to have the target separated from the data
# used to train our predictive model.

# %%
target_name = "class"
target = df[target_name]
data = df.drop(columns=target_name)

# %% [markdown]
# We start by selecting only the numerical columns as seen in the previous
# notebook.

# %%
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

# %% [markdown]
# Next, let's split the data and target into a train and test set.

# %%
from sklearn.model_selection import train_test_split

data_numeric_train, data_numeric_test, target_train, target_test = \
    train_test_split(data_numeric, target, random_state=0)

# %% [markdown]
# We will first create as dummy classifier which will always predict the
# high revenue class class, i.e. `" >50K"`, and check the performance.

# %%
from sklearn.dummy import DummyClassifier

class_to_predict = " >50K"
high_revenue_clf = DummyClassifier(strategy="constant",
                                   constant=class_to_predict)
high_revenue_clf.fit(data_numeric_train, target_train)
score = high_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting only high revenue: {score:.3f}")

# %% [markdown]
# We clearly see that the score is below 0.5 which might be surprising at
# first. We will now check the performance of a model which always predict the
# low revenue class, i.e. `" <=50K"`.

# %%
class_to_predict = " <=50K"
low_revenue_clf = DummyClassifier(strategy="constant",
                                  constant=class_to_predict)
low_revenue_clf.fit(data_numeric_train, target_train)
score = low_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting only low revenue: {score:.3f}")

# %% [markdown]
# We observe that this model as an accuracy higher than 0.5. This due to the
# fact that we have 3/4 of the target belonging to low-revenue class.

# %% [markdown]
# Therefore, any predictive model giving results below this dummy classifier
# will not be helpful.

# %%
df["class"].value_counts()

# %%
(target == " <=50K").mean()

# %% [markdown]
# In practice, we could have the strategy `"most_frequent"` to predict the
# class that appears the most in the training target.

# %%
most_freq_revenue_clf = DummyClassifier(strategy="most_frequent")
most_freq_revenue_clf.fit(data_numeric_train, target_train)
score = most_freq_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting the most frequent class: {score:.3f}")

# %% [markdown]
# So 81% accuracy is significantly better than 76% which is the score of a
# baseline model that would always predict the most frequent class which is the
# low revenue class: `" <=50K"`.
