# %% [markdown]
# # 📃 Solution fo Exercise 01
#
# The aim of this exercise is to make the following experiments:
#
# * train and test a support vector machine classifier through
#   cross-validation;
# * study the effect of the parameter gamma of this classifier using a
#   validation curve;
# * study if it would be useful in term of classification if we could add new
#   samples in the dataset using a learning curve.
#
# To make these experiments we will first load the blood transfusion dataset.

# %%
import pandas as pd

data = pd.read_csv("../datasets/blood_transfusion.csv")
X, y = data.drop(columns="Class"), data["Class"]

# %% [markdown]
# Create a machine learning pipeline which will standardize the data and then
# use a support vector machine with an RBF kernel

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC())

# %% [markdown]
# Evaluate the performance of the previous model by cross-validation with a
# `ShuffleSplit` scheme.

# %%
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(random_state=0)
cv_results = cross_validate(model, X, y, cv=cv, n_jobs=-1)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
print(
    f"Accuracy score of our model:\n"
    f"{cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

# %% [markdown]
# The parameter gamma is one of the parameter controlling under-/over-fitting
# in support vector machine with an RBF kernel. Compute the validation curve
# to evaluate the effect of the parameter gamma. You can make vary the value
# of the parameter gamma between `10e-3` and `10e2` by generating samples on
# log scale.

# %%
import numpy as np
from sklearn.model_selection import validation_curve

gammas = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
train_scores, test_scores = validation_curve(
    model, X, y, param_name=param_name, param_range=gammas, cv=cv, n_jobs=-1)

# %% [markdown]
# Plot the validation curve for the train and test scores.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

_, ax = plt.subplots()

for name, scores in zip(
    ["Empirical score", "Generalization score"], [train_scores, test_scores]
):
    ax.plot(
        gammas, scores.mean(axis=1), linestyle="-.", label=name,
        alpha=0.8)
    ax.fill_between(
        gammas, scores.mean(axis=1) - scores.std(axis=1),
        scores.mean(axis=1) + scores.std(axis=1),
        alpha=0.5, label=f"std. dev. {name.lower()}")

ax.set_xticks(gammas)
ax.set_xscale("log")
ax.set_xlabel("Value of hyperparameter $\gamma$")
ax.set_ylabel("Accuracy score")
ax.set_title("Validation score of support vector machine")
_ = plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")

# %% [markdown]
# Looking at the curve, we can clearly identify the over-fitting regime of
# the SVC classifier: when `gamma > 1`. The best setting is around `gamma = 1`
# while for `gamma < 1`, it is not very clear that the classifier is
# under-fitting but the generalization score is worse than for `gamma = 1`.
#
# Now, you can make an analysis to check if adding new samples to the dataset
# could help our model to better generalize. Compute the learning curve by
# computing the train and test score for different training dataset size.
# Plot the train and test score in respect with the number of samples.

# %%
from sklearn.model_selection import learning_curve

train_sizes = np.linspace(0.1, 1, num=10)
results = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1)
train_size, train_scores, test_scores = results[:3]

# %%
_, ax = plt.subplots()

for name, scores in zip(
    ["Empirical score", "Generalization score"], [train_scores, test_scores]
):
    ax.plot(
        train_sizes, scores.mean(axis=1), linestyle="-.", label=name,
        alpha=0.8)
    ax.fill_between(
        train_sizes, scores.mean(axis=1) - scores.std(axis=1),
        scores.mean(axis=1) + scores.std(axis=1),
        alpha=0.5, label=f"std. dev. {name.lower()}")

ax.set_xticks(train_sizes)
ax.set_xlabel("Number of samples in the training set")
ax.set_ylabel("Accuracy")
ax.set_title("Learning curve for support vector machine")
_ = plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")

# %% [markdown]
# We observe that adding new samples in the dataset does not improve the
# generalization score. We can only conclude that the standard deviation of
# the empirical error is decreasing when adding more samples which is not a
# surprise.
