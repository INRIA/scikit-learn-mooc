# %% [markdown]
# # 📃 Solution for Exercise 01
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

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]

# %% [markdown]
# We will use a support vector machine classifier (SVM). In its most simple
# form, a SVM classifier is a linear classifier behaving similarly to a
# logistic regression. Indeed, the optimization used to find the optimal
# weights of the linear model are different but we don't need to know these
# details for the exercise.
#
# Also, this classifier can become more flexible/expressive by using a
# so-called kernel. The model becomes non-linear. Again, no requirement
# regarding the mathematics is required to accomplish this exercise.
#
# We will use an RBF kernel where a parameter `gamma` allows to tune the
# flexibility of the model.
#
# First let's create a predictive pipeline made of:
#
# * a [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
#   with default parameter;
# * a [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
#   where the parameter `kernel` could be set to `"rbf"`. Note that this is the
#   default.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC())

# %% [markdown]
# Evaluate the statistical performance of your model by cross-validation with a
# `ShuffleSplit` scheme. Thus, you can use
# [`sklearn.model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
# and pass a [`sklearn.model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
# to the `cv` parameter. Only fix the `random_state=0` in the `ShuffleSplit`
# and let the other parameters to the default.

# %%
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(random_state=0)
cv_results = cross_validate(model, data, target, cv=cv, n_jobs=-1)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
print(
    f"Accuracy score of our model:\n"
    f"{cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

# %% [markdown]
# As previously mentioned, the parameter `gamma` is one of the parameter
# controlling under/over-fitting in support vector machine with an RBF kernel.
#
# Compute the validation curve to evaluate the effect of the parameter `gamma`.
# You can vary its value between `10e-3` and `10e2` by generating samples on a
# logarithmic scale. Thus, you can use `np.logspace(-3, 2, num=30)`.
#
# Since we are manipulating a `Pipeline` the parameter name will be set to
# `svc__gamma` instead of only `gamma`. You can retrieve the parameter name
# using `model.get_params().keys()`. We will go more into details regarding
# accessing and setting hyperparameter in the next section.

# %%
import numpy as np
from sklearn.model_selection import validation_curve

gammas = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
train_scores, test_scores = validation_curve(
    model, data, target, param_name=param_name, param_range=gammas, cv=cv,
    n_jobs=-1)

# %% [markdown]
# Plot the validation curve for the train and test scores.

# %%
import matplotlib.pyplot as plt

_, ax = plt.subplots()

for name, scores in zip(
    ["Training score", "Testing score"], [train_scores, test_scores]
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
ax.set_xlabel(r"Value of hyperparameter $\gamma$")
ax.set_ylabel("Accuracy score")
ax.set_title("Validation score of support vector machine")
_ = plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")

# %% [markdown]
# Looking at the curve, we can clearly identify the over-fitting regime of
# the SVC classifier when `gamma > 1`.
# The best setting is around `gamma = 1` while for `gamma < 1`,
# it is not very clear if the classifier is under-fitting but the
# testing score is worse than for `gamma = 1`.

# %% [markdown]
# Now, you can perform an analysis to check whether adding new samples to the
# dataset could help our model to better generalize. Compute the learning curve
# (using [`sklearn.model_selection.learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html))
# by computing the train and test scores for different training dataset size.
# Plot the train and test scores with respect to the number of samples.

# %%
from sklearn.model_selection import learning_curve

train_sizes = np.linspace(0.1, 1, num=10)
results = learning_curve(
    model, data, target, train_sizes=train_sizes, cv=cv, n_jobs=-1)
train_size, train_scores, test_scores = results[:3]

# %%
_, ax = plt.subplots()

for name, scores in zip(
    ["Training score", "Testing score"], [train_scores, test_scores]
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
# testing score. We can only conclude that the standard deviation of
# the training error is decreasing when adding more samples which is not a
# surprise.
