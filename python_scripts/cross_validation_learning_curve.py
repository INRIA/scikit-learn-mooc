# %% [markdown]
# # Effect of the sample size in cross-validation
#
# In the previous notebook, we presented the general cross-validation framework
# and how to assess if a predictive model is underfiting, overfitting, or
# generalizing. Besides these aspects, it is also important to understand how
# the different errors are influenced by the number of samples available.
#
# In this notebook, we will show this aspect by looking a the
# variability of the different errors.
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
#
# ## Ability of a model to learn depending of the sample size
#
# We recall that the size of the dataset is given by the number
# of rows in `X` / the length of the vector `y`.

# %%
target.size

# %% [markdown]
# Let's do an experiment and reduce the number of samples and repeat the
# previous experiment. We will create a function that define a `ShuffleSplit`
# and given a regressor and the data `X` and `y` will run a cross-validation.
# The function will finally return the generalization error as a NumPy array.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit


def make_cv_analysis(regressor, data, target):
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    cv_results = cross_validate(regressor, data, target,
                                cv=cv, scoring="neg_mean_absolute_error",
                                return_train_score=True)
    cv_results = pd.DataFrame(cv_results)
    return (cv_results["test_score"] * -1).values


# %% [markdown]
# Now that we have a function to run each experiment, we will create an array
# defining the number of samples for which we want to run the experiments.

# %%
sample_sizes = [100, 500, 1000, 5000, 10000, 15000, target.size]

# %%
import numpy as np

# to make our results reproducible
rng = np.random.RandomState(0)

# create a dictionary where we will store the result of each run
scores_sample_sizes = {"# samples": [], "test error": []}
for n_samples in sample_sizes:
    # select a subset of the data with a specific number of samples
    sample_idx = rng.choice(
        np.arange(target.size), size=n_samples, replace=False)
    data_sampled, target_sampled = data.iloc[sample_idx], target[sample_idx]
    # run the experiment
    score = make_cv_analysis(regressor, data_sampled, target_sampled)
    # store the results
    scores_sample_sizes["# samples"].append(n_samples)
    scores_sample_sizes["test error"].append(score)

# %% [markdown]
# Now, we collected all our results and we will create a pandas dataframe to
# easily make some plot.

# %%
scores_sample_sizes = pd.DataFrame(
    np.array(scores_sample_sizes["test error"]).T,
    columns=scores_sample_sizes["# samples"],
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

sns.displot(scores_sample_sizes, kind="kde")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Generalization errors distribution \n"
              "by varying the sample size")

# %% [markdown]
# For the different sample sizes, we plotted the distribution of the
# generalization error. We observe that the smaller the number of samples is,
# the larger the variance of the generalization errors is. Thus, having a small
# number of samples might put us in a situation where it is impossible to get a
# reliable evaluation.
#
# ## Learning curve
#
# Here, we plotted the different curves to highlight the issue of small sample
# size. However, this experiment is also used to draw the so-called **learning
# curve**. This curve gives some additional indication regarding the benefit of
# adding new training samples to improve a model's performance.

# %%
from sklearn.model_selection import learning_curve

cv = ShuffleSplit(n_splits=30, test_size=0.2)
results = learning_curve(
    regressor, data, target, train_sizes=sample_sizes[:-1], cv=cv,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_size, train_scores, test_scores = results[:3]
train_errors, test_errors = -train_scores, -test_scores

# %% [markdown]
# Now, we can plot the curve curve.

# %%
_, ax = plt.subplots()

error_type = ["Empirical error", "Generalization error"]
errors = [train_errors, test_errors]

for name, err in zip(error_type, errors):
    ax.plot(train_size, err.mean(axis=1), linestyle="-.", label=name,
            alpha=0.8)
    ax.fill_between(train_size, err.mean(axis=1) - err.std(axis=1),
                    err.mean(axis=1) + err.std(axis=1),
                    alpha=0.5, label=f"std. dev. {name.lower()}")

ax.set_xticks(train_size)
ax.set_xscale("log")
ax.set_xlabel("Number of samples in the training set")
ax.set_ylabel("Mean absolute error (k$)")
ax.set_title("Learning curve for decision tree")
_ = plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")

# %% [markdown]
# We see that the more samples we add to the training set on this learning
# curve, the lower the error becomes. With this curve, we are searching for the
# plateau for which there is no benefit to adding samples anymore or assessing
# the potential gain of adding more samples into the training set.
#
# For this dataset we notice that our decision tree model would really benefit
# from additional datapoints to reduce the amount of over-fitting and hopefully
# reduce the generalization error even further.
#
# ## Summary
#
# In the notebook, we learnt:
#
# * the influence of the number of samples in a dataset, especially on the
#   variability of the errors reported when running the cross-validation;
# * about the learning curve that is a visual representation of the capacity
#   of a model to improve by adding new samples.
