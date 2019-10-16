# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solution for Exercise 01
#
# The goal of is to compare the performance of our classifier to some baseline classifier that  would ignore the input data and instead make constant predictions:

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# %%
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])
numerical_columns = [
    c for c in data.columns if data[c].dtype.kind in ["i", "f"]]
data_numeric = data[numerical_columns]

# %%
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

high_revenue_clf = DummyClassifier(strategy="constant",
                                   constant=" >50K")
scores = cross_val_score(high_revenue_clf, data_numeric, target)
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %%
low_revenue_clf = DummyClassifier(strategy="constant",
                                  constant=" <=50K")
scores = cross_val_score(low_revenue_clf, data_numeric, target)
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %%
most_freq_revenue_clf = DummyClassifier(strategy="most_frequent")
scores = cross_val_score(most_freq_revenue_clf, data_numeric, target)
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# So 81% accuracy is significantly better than 76% which is the score of a baseline model that would always predict the most frequent class which is the low revenue class: `" <=50K"`.
#
# In this dataset, we can see that the target classes are imbalanced: almost 3/4 of the records are people with a revenue below 50K:

# %%
df["class"].value_counts()

# %%
(target == " <=50K").mean()
