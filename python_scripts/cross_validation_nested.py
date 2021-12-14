# %% [markdown]
# # Nested cross-validation
#
# Cross-validation can be used both for hyperparameter tuning and for
# estimating the generalization performance of a model. However, using
# it for both purpose at the same time is problematic, as the resulting
# evaluation is likely to underestimate some overfitting that results from
# the hyperparameter tuning procedure itself.
#
# Philosophically, hyperparameter tuning is a form of machine learning
# itself and therefore, we need another outer loop of cross-validation to
# properly evaluate the generalization performance of the full modeling
# procedure.
#
# This notebook introduces nested cross-validation and highlights its impact
# on the estimated generalization performance compared to naively using
# a single level of cross-validation, both for hyperparameter tuning and
# evaluation of the generalization performance.
#
# We will illustrate this difference using the breast cancer dataset.

# %%
from sklearn.datasets import load_breast_cancer

data, target = load_breast_cancer(return_X_y=True)

# %% [markdown]
# Now, we'll make a minimal example using the utility `GridSearchCV` to find
# the best parameters via cross-validation.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {"C": [0.1, 1, 10], "gamma": [.01, .1]}
model_to_tune = SVC()

search = GridSearchCV(
    estimator=model_to_tune, param_grid=param_grid, n_jobs=2
)
search.fit(data, target)

# %% [markdown]
# We recall that `GridSearchCV` trains a model with some specific parameters
# on a training set and evaluate it on a testing set. This evaluation is
# done via cross-validation using the `cv` parameter. The procedure is then
# repeated for all possible combinations of parameters given in `param_grid`.
#
# The attribute `best_params_` gives us the best set of parameters that
# maximize the mean score on the internal test sets.

# %%
print(f"The best parameters found are: {search.best_params_}")

# %% [markdown]
# We can also show the mean score obtained by using the parameters `best_params_`.

# %%
print(f"The mean CV score of the best model is: {search.best_score_:.3f}")

# %% [markdown]
# At this stage, one should be extremely careful using this score. The
# misinterpretation would be the following: since the score was computed on a
# test set, it could be considered our model's testing score.
#
# However, we should not forget that we used this score to pick-up the best
# model. It means that we used knowledge from the test set (i.e. test score) to
# decide our model's training parameter.
#
# Thus, this score is not a reasonable estimate of our testing error.
# Indeed, we can show that it will be too optimistic in practice. The good way
# is to use a "nested" cross-validation. We will use an inner cross-validation
# corresponding to the previous procedure shown to optimize the
# hyperparameters. We will also include this procedure within an outer
# cross-validation, which will be used to estimate the testing error of
# our tuned model.
#
# In this case, our inner cross-validation will always get the training set of
# the outer cross-validation, making it possible to compute the testing
# score on a completely independent set.
#
# We will show below how we can create such nested cross-validation and obtain
# the testing score.

# %%
from sklearn.model_selection import cross_val_score, KFold

# Declare the inner and outer cross-validation
inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

# Inner cross-validation for parameter search
model = GridSearchCV(
    estimator=model_to_tune, param_grid=param_grid, cv=inner_cv, n_jobs=2)

# Outer cross-validation to compute the testing score
test_score = cross_val_score(model, data, target, cv=outer_cv, n_jobs=2)
print(f"The mean score using nested cross-validation is: "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %% [markdown]
# In the example above, the reported score is more trustworthy and should be close
# to production's expected generalization performance.
#
# We now illustrate the difference between the nested and non-nested
# cross-validation scores to show that the latter are too optimistic in
# practice. In this regard, we repeat the experiment several times and
# shuffle the data differently to ensure that our conclusion does not depend on
# a particular resampling of the data.

# %%
test_score_not_nested = []
test_score_nested = []

N_TRIALS = 20
for i in range(N_TRIALS):
    # For each trial, we use cross-validation splits on independently
    # randomly shuffled data by passing distinct values to the random_state
    # parameter.
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    model = GridSearchCV(estimator=model_to_tune, param_grid=param_grid,
                         cv=inner_cv, n_jobs=2)
    model.fit(data, target)
    test_score_not_nested.append(model.best_score_)

    # Nested CV with parameter optimization
    test_score = cross_val_score(model, data, target, cv=outer_cv, n_jobs=2)
    test_score_nested.append(test_score.mean())

# %% [markdown]
# We can merge the data together and make a box plot of the two strategies.

# %%
import pandas as pd

all_scores = {
    "Not nested CV": test_score_not_nested,
    "Nested CV": test_score_nested,
}
all_scores = pd.DataFrame(all_scores)

# %%
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
all_scores.plot.box(color=color, vert=False)
plt.xlabel("Accuracy")
_ = plt.title("Comparison of mean accuracy obtained on the test sets with\n"
              "and without nested cross-validation")

# %% [markdown]
# We observe that the generalization performance estimated without using
# nested CV is higher than what we obtain with nested CV. The reason is that
# the tuning procedure itself selects the model with the highest inner CV score.
# If there are many hyper-parameter combinations and the inner CV scores have
# comparatively large standard deviations, taking the maximum value can lure
# the naive data scientist into over-estimating the true generalization performance
# of the result of the full learning procedure. By using an outer cross-validation
# procedure one gets a more trustworthy estimate of the generalization performance
# of the full learning procedure, including the effect of tuning the hyperparameters.
#
# As a conclusion, when optimizing parts of the machine learning pipeline (e.g.
# hyperparameter, transform, etc.), one needs to use nested cross-validation to
# evaluate the generalization performance of the predictive model. Otherwise,
# the results obtained without nested cross-validation are often overly optimistic.
