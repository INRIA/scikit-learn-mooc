# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M2.01
#
# The aim of this exercise is to make the following experiments:
#
# * train and test a support vector machine classifier through cross-validation;
# * study the effect of the parameter gamma of this classifier using a
#   validation curve;
# * use a learning curve to determine the usefulness of adding new samples in
#   the dataset when building a classifier.
#
# To make these experiments we will first load the blood transfusion dataset.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]

# %% [markdown]
# We will use a support vector machine classifier (SVM). In its most simple
# form, a SVM classifier is a linear classifier behaving similarly to a logistic
# regression. Indeed, the optimization used to find the optimal weights of the
# linear model are different but we don't need to know these details for the
# exercise.
#
# Also, this classifier can become more flexible/expressive by using a so-called
# kernel that makes the model become non-linear. Again, no requirement regarding
# the mathematics is required to accomplish this exercise.
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
# Write your code here.

# %% [markdown]
# Evaluate the generalization performance of your model by cross-validation with
# a `ShuffleSplit` scheme. Thus, you can use
# [`sklearn.model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
# and pass a
# [`sklearn.model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
# to the `cv` parameter. Only fix the `random_state=0` in the `ShuffleSplit` and
# let the other parameters to the default.

# %%
# Write your code here.

# %% [markdown]
# As previously mentioned, the parameter `gamma` is one of the parameters
# controlling under/over-fitting in support vector machine with an RBF kernel.
#
# Evaluate the effect of the parameter `gamma` by using the
# [`sklearn.model_selection.validation_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)
# function. You can leave the default `scoring=None` which is equivalent to
# `scoring="accuracy"` for classification problems. You can vary `gamma` between
# `10e-3` and `10e2` by generating samples on a logarithmic scale with the help
# of `np.logspace(-3, 2, num=30)`.
#
# Since we are manipulating a `Pipeline` the parameter name will be set to
# `svc__gamma` instead of only `gamma`. You can retrieve the parameter name
# using `model.get_params().keys()`. We will go more into detail regarding
# accessing and setting hyperparameter in the next section.

# %%
# Write your code here.

# %% [markdown]
# Plot the validation curve for the train and test scores.

# %%
# Write your code here.

# %% [markdown]
# Now, you can perform an analysis to check whether adding new samples to the
# dataset could help our model to better generalize. Compute the learning curve
# (using [`sklearn.model_selection.learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html))
# by computing the train and test scores for different training dataset size.
# Plot the train and test scores with respect to the number of samples.

# %%
# Write your code here.
