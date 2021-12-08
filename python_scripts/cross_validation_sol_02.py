# %% [markdown]
# # 📃 Solution for Exercise M7.01
#
# This notebook aims at building some baseline classifiers, which we use as
# references to assess the relative predictive performance of a given model of
# interest.
#
# We illustrate those baselines with the help of the Adult Census dataset,
# using only the numerical features for the sake of simplicity.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric-all.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]

# %% [markdown]
# First, define a `ShuffleSplit` cross-validation strategy taking half of the
# samples as a testing at each round.

# %%
# solution
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

# %% [markdown]
# Next, create a machine learning pipeline composed of a transformer to
# standardize the data followed by a logistic regression classifier.

# %%
# solution
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

classifier = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# Compute the cross-validation (test) scores for the classifier on this
# dataset. Keep the results in a numpy array or a pandas Series.

# %%
# solution
from sklearn.model_selection import cross_validate

cv_results_logistic_regression = cross_validate(
    classifier, data, target, cv=cv, n_jobs=2
)

test_score_logistic_regression = pd.Series(
    cv_results_logistic_regression["test_score"], name="Logistic Regression"
)


# %% [markdown]
# Now, compute the cross-validation scores of a dummy classifier that
# constantly predicts the most frequent class observed the training set. Please
# refer to the online documentation for the [sklearn.dummy.DummyClassifier
# ](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
# class.

# %%
# solution
from sklearn.dummy import DummyClassifier

most_frequent_classifier = DummyClassifier(strategy="most_frequent")
cv_results_most_frequent = cross_validate(
    most_frequent_classifier, data, target, cv=cv, n_jobs=2
)
test_score_most_frequent = pd.Series(
    cv_results_most_frequent["test_score"],
    name="Most frequent class predictor"
)

# %% [markdown]
# Now that we collected the results from the baselines and the model, plot
# the distributions of the different test scores.

# %% [markdown]
# We concatenate the different test score in the same pandas dataframe.

# %%
# solution
final_test_scores = pd.concat(
    [test_score_logistic_regression, test_score_most_frequent],
    axis=1,
)

# %% [markdown]
# Next, plot the distributions of the test scores.

# %%
# solution
import matplotlib.pyplot as plt

final_test_scores.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Accuracy (%)")
_ = plt.title("Distribution of the test scores")

# %% [markdown] tags=["solution"]
# We observe that the dummy classifier with the strategy `most_frequent` is
# equivalent to the permutation score. We can also conclude that our model
# is better than the other baseline.

# %% [markdown]
# Change the strategy of the dummy classifier to `stratified`, compute the
# results and plot the distribution together with the other results. Explain
# why the results get worse.

# %%
# solution
dummy = DummyClassifier(strategy="stratified")
result_dummy_stratify = cross_validate(dummy, data, target, cv=cv, n_jobs=2)
test_score_dummy_stratify = pd.Series(
    result_dummy_stratify["test_score"], name="Dummy 'stratify' score"
)

# %% tags=["solution"]
final_test_scores = pd.concat(
    [
        test_score_classifier,
        test_score_permutation,
        test_score_dummy,
        test_score_dummy_stratify,
    ],
    axis=1,
)

# %% tags=["solution"]
final_test_scores.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Accuracy (%)")
_ = plt.title("Distribution of the test scores")

# %% [markdown] tags=["solution"]
# We see that using `strategy="stratified"`, the results are much worse than
# with the `most_frequent` strategy. Since the classes are imbalanced,
# predicting the most frequent involves that we will be right for the
# proportion of this class (~75% of the samples). However, the `"stratified"`
# strategy will randomly generate predictions by respecting the training
# set's class distribution, resulting in some wrong predictions even for
# the most frequent class, hence we obtain a lower accuracy.
#
# Please refer to the scikit-learn documentation on [sklearn.dummy.DummyClassifier
# ](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
# for more details on the meaning of the `strategy="stratified"` parameter.

# %% tags=["solution"]
from sklearn.model_selection import cross_val_score

dummy_models = {
    "Dummy 'most_frequent'": DummyClassifier(strategy="most_frequent"),
    "Dummy 'stratified'": DummyClassifier(strategy="stratified"),
}
n_runs = 3

for run_idx in range(n_runs):
    final_scores = pd.DataFrame(
        {
            f"{name} score": cross_val_score(model, data, target, cv=cv, n_jobs=2)
            for name, model in dummy_models.items()
        }
    )

    final_scores.plot.hist(bins=50, density=True, edgecolor="black")
    plt.legend(bbox_to_anchor=(1.05, 0.8))
    plt.xlabel("Accuracy (%)")
    _ = plt.title(f"Distribution of scores in run #{run_idx}")
