# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solution for Exercise 01
#
# The goal of is to compare the performance of our classifier to some baseline
# classifier that would ignore the input data and instead make constant
# predictions:

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# %%
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_include=["int", "float"])
numerical_columns = numerical_columns_selector(data)
data_numeric = data[numerical_columns]

# %%
from sklearn.model_selection import train_test_split

data_numeric_train, data_numeric_test, target_train, target_test = \
    train_test_split(data_numeric, target, random_state=0)

# %%
from sklearn.dummy import DummyClassifier

high_revenue_clf = DummyClassifier(strategy="constant",
                                   constant=" >50K")
high_revenue_clf.fit(data_numeric_train, target_train)
score = high_revenue_clf.score(data_numeric_test, target_test)
print(f"{score:.3f}")

# %%
low_revenue_clf = DummyClassifier(strategy="constant",
                                  constant=" <=50K")
low_revenue_clf.fit(data_numeric_train, target_train)
score = low_revenue_clf.score(data_numeric_test, target_test)
print(f"{score:.3f}")


# %%
most_freq_revenue_clf = DummyClassifier(strategy="most_frequent")
most_freq_revenue_clf.fit(data_numeric_train, target_train)
score = most_freq_revenue_clf.score(data_numeric_test, target_test)
print(f"{score:.3f}")

# %% [markdown]
# So 81% accuracy is significantly better than 76% which is the score of a
# baseline model that would always predict the most frequent class which is the
# low revenue class: `" <=50K"`.
#
# In this dataset, we can see that the target classes are imbalanced: almost
# 3/4 of the records are people with a revenue below 50K:

# %%
df["class"].value_counts()

# %%
(target == " <=50K").mean()
