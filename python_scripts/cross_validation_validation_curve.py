# %% [markdown]
# # Overfit-generalization-underfit
#
# In the previous notebook, we presented the general cross-validation framework
# and how it helps us quantify the empirical and generalization errors as well
# as their fluctuations.
#
# In this notebook, we will put these two errors into perspective and show how
# they can help us know if our model generalizes, overfit, or underfit.
#
# Let's first load the data and create the same model as in the previous
# notebook.

# %%
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target

# %%
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

# %% [markdown]
# ## Overfitting vs. underfitting
#
# To better understand the performance of our model and maybe find insights on
# how to improve it, we will compare the generalization error with the
# empirical error. Thus, we need to compute the error on the training set,
# which is possible using the `cross_validate` function.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)
cv_results = cross_validate(regressor, data, target,
                            cv=cv, scoring="neg_mean_absolute_error",
                            return_train_score=True, n_jobs=2)
cv_results = pd.DataFrame(cv_results)

# %% [markdown]
# We will select the train and test score and take the error instead.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

scores = pd.DataFrame()
scores[["train error", "test error"]] = -cv_results[
    ["train_score", "test_score"]]
sns.histplot(scores, bins=50)
_ = plt.xlabel("Mean absolute error (k$)")

# %% [markdown]
# By plotting the distribution of the empirical and generalization errors, we
# get information about whether our model is over-fitting, under-fitting (or
# both at the same time).
#
# Here, we observe a **small empirical error** (actually zero), meaning that
# the model is **not under-fitting**: it is flexible enough to capture any
# variations present in the training set.
#
# However the **significantly larger generalization error** tells us that the
# model is **over-fitting**: the model has memorized many variations of the
# training set that could be considered "noisy" because they do not generalize
# to help us make good prediction on the test set.
#
# ## Validation curve

#
# Some model hyperparameters are usually the key to go from a model that
# underfits to a model that overfits, hopefully going through a region were we
# can get a good balance between the two. We can acquire knowledge by plotting
# a curve called the validation curve. This curve applies the above experiment
# and varies the value of a hyperparameter.
#
# For the decision tree, the `max_depth` parameter is used to control the
# tradeoff between under-fitting and over-fitting.

# %%
# %%time
from sklearn.model_selection import validation_curve

max_depth = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    regressor, data, target, param_name="max_depth", param_range=max_depth,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
train_errors, test_errors = -train_scores, -test_scores

# %% [markdown]
# Now that we collected the results, we will show the validation curve by
# plotting the training and testing errors (as well as their deviations).

# %%
_, ax = plt.subplots()

error_type = ["Training error", "Testing error"]
errors = [train_errors, test_errors]

for name, err in zip(error_type, errors):
    ax.plot(max_depth, err.mean(axis=1), linestyle="-.", label=name,
            alpha=0.8)
    ax.fill_between(max_depth, err.mean(axis=1) - err.std(axis=1),
                    err.mean(axis=1) + err.std(axis=1), alpha=0.5,
                    label=f"std. dev. {name.lower()}")

ax.set_xticks(max_depth)
ax.set_xlabel("Maximum depth of decision tree")
ax.set_ylabel("Mean absolute error (k$)")
ax.set_title("Validation curve for decision tree")
_ = plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")

# %% [markdown]
# The validation curve can be divided into three areas:
#
# - For `max_depth < 10`, the decision tree underfits. The training error and
#   therefore the testing error are both high. The model is too
#   constrained and cannot capture much of the variability of the target
#   variable.
#
# - The region around `max_depth = 10` corresponds to the parameter for which
#   the decision tree generalizes the best. It is flexible enough to capture a
#   fraction of the variability of the target that generalizes, while not
#   memorizing all of the noise in the target.
#
# - For `max_depth > 10`, the decision tree overfits. The training error
#   becomes very small, while the testing error increases. In this
#   region, the models create decision specifically for noisy samples harming
#   its ability to generalize to test data.
#
# Note that for `max_depth = 10`, the model overfits a bit as there is a gap
# between the training error and the testing error. It can also
# potentially underfit also a bit at the same time, because the training error
# is still far from zero (more than 30 k\$), meaning that the model might
# still be too constrained to model interesting parts of the data. However the
# testing error is minimal, and this is what really matters. This is the
# best compromise we could reach by just tuning this parameter.
#
# We were lucky that the variance of the errors was small compared to their
# respective values, and therefore the conclusions above are quite clear. This
# is not necessarily always the case.
#
# ## Summary:
#
# In this notebook, we saw:
#
# * how to identify whether a model is generalizing, overfitting, or
#   underfitting;
# * how to check influence of an hyperparameter on the tradeoff
#   underfit/overfit.
