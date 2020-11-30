# %% [markdown]
# # 📃 Solution for Exercise 02
#
# This notebook aims at building baselines classifier with which we can compare
# our predictive model. Besides, we will check the differences with the
# baselines that we saw in regression.
#
# We will use the adult census dataset but only the numerical features.

# %%
import pandas as pd

data = pd.read_csv("../datasets/adult-census-numeric-all.csv")
X, y = data.drop(columns="class"), data["class"]

# %% [markdown]
# First, define a `ShuffleSplit` cross-validation strategy taking half of the
# sample as a testing at each round.

# %%
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

# %% [markdown]
# Next, create a machine learning pipeline composed of a transformer to
# standardize the data followed by a logistic regression.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

classifier = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# Get the test score by using the model, the data, and the cross-validation
# strategy that you defined above.

# %%
from sklearn.model_selection import cross_validate

result_classifier = cross_validate(classifier, X, y, cv=cv, n_jobs=-1)

test_score_classifier = pd.Series(
    result_classifier["test_score"], name="Classifier score")

# %% [markdown]
# Using the `sklearn.model_selection.permutation_test_score` function,
# check the chance level of the previous model.

# %%
from sklearn.model_selection import permutation_test_score

score, permutation_score, pvalue = permutation_test_score(
    classifier, X, y, cv=cv, n_jobs=-1, n_permutations=10)
test_score_permutation = pd.Series(permutation_score, name="Permuted score")

# %% [markdown]
# Finally, compute the test score of a dummy classifier which would predict
# the most frequent class from the training set. You can look at the
# `sklearn.dummy.DummyClassifier` class.

# %%
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
result_dummy = cross_validate(dummy, X, y, cv=cv, n_jobs=-1)
test_score_dummy = pd.Series(result_dummy["test_score"], name="Dummy score")

# %% [markdown]
# Now that we collected the results from the baselines and the model, plot
# the distributions of the different test scores.

# %% [markdown]
# We concatenate the different test score in the same pandas dataframe.

# %%
final_test_scores = pd.concat(
    [test_score_classifier, test_score_permutation, test_score_dummy],
    axis=1,
)

# %% [markdown]
# And finally we plot the distributions of the test scores.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

sns.displot(final_test_scores, kind="kde")
_ = plt.xlabel("Accuracy (%)")

# %% [markdown]
# We observe that the dummy classifier with the strategy `most_frequent` is
# equivalent to use the permutation score. We also can conclude that our model
# is better than the other baseline.

# %% [markdown]
# Change the strategy of the dummy classifier to `stratified`, compute the
# results and plot the distribution together with the other results. Explain
# why the results get worse.

# %%
dummy = DummyClassifier(strategy="stratified")
result_dummy_stratify = cross_validate(dummy, X, y, cv=cv, n_jobs=-1)
test_score_dummy_stratify = pd.Series(
    result_dummy_stratify["test_score"], name="Dummy 'stratify' score")

# %%
final_test_scores = pd.concat(
    [
        test_score_classifier, test_score_permutation,
        test_score_dummy, test_score_dummy_stratify,
    ],
    axis=1,
)

# %%
sns.displot(final_test_scores, kind="kde")
_ = plt.xlabel("Accuracy (%)")

# %% [markdown]
# We see that using `strategy="stratified"`, the results are much worse than
# with the `most_frequent` strategy. Since the classes are imbalanced,
# predicting the most frequent involves that we will be right for the
# proportion of this class (~75% of the samples). However, by using the
# `stratified` strategy, wrong preditions will be made even for the most
# frequent class, hence we obtain a lower accuracy.
