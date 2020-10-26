# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature selection
#

# %% [markdown]
# ## Benefit of feature selection in practice
#
# ### Speed-up train and scoring time
# The principal advantage of selecting features within a machine learning
# pipeline is to reduce the time to train this pipeline and its time to
# predict. We will give an example to highlights these advantages. First, we
# generate a synthetic dataset to control the number of features that will be
# informative, redundant, repeated, and random.

# %%
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=5000,
    n_features=100,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=0,
)

# %% [markdown]
# We chose to create a dataset with two informative features among a hundred.
# To simplify our example, we did not include either redundant or repeated
# features.
#
# We will create two machine learning pipelines. The former will be a random
# forest that will use all available features. The latter will also be a random
# forest, but we will add a feature selection step to train this classifier.
# The feature selection is based on a univariate test (ANOVA F-value) between
# each feature and the target that we want to predict. The features with the
# two most significant scores are selected.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import make_pipeline

model_without_selection = RandomForestClassifier(n_jobs=-1)
model_with_selection = make_pipeline(
    SelectKBest(score_func=f_classif, k=2),
    RandomForestClassifier(n_jobs=-1),
)

# %% [markdown]
# We will measure the average time spent to train each pipeline and make it
# predict. Besides, we will compute the generalization score of the model. We
# will collect these results via cross-validation.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results_without_selection = pd.DataFrame(
    cross_validate(model_without_selection, X, y)
)
cv_results_with_selection = pd.DataFrame(
    cross_validate(model_with_selection, X, y, return_estimator=True),
)

# %%
cv_results = pd.concat(
    [cv_results_without_selection, cv_results_with_selection],
    axis=1,
    keys=["Without feature selection", "With feature selection"],
).swaplevel(axis="columns")

# %% [markdown]
# Let's first analyze the train and score time for each pipeline.

# %%
import matplotlib.pyplot as plt

cv_results["fit_time"].plot.box(vert=False, whis=100)
plt.xlabel("Elapsed time (s)")
_ = plt.title("Time to fit the model")

# %%
cv_results["score_time"].plot.box(vert=False, whis=100)
plt.xlabel("Elapsed time (s)")
_ = plt.title("Time to make prediction")

# %% [markdown]
# We can draw the same conclusions for both training and scoring elapsed time:
# selecting the most informative features speed-up our pipeline.
#
# Of course, such speed-up is beneficial only if the performance in terms of
# metrics remain the same. Let's check the generalization score.

# %%
cv_results["test_score"].plot.box(vert=False, whis=100)
plt.xlabel("Accuracy score")
_ = plt.title("Test score via cross-validation")

# %% [markdown]
# We can observe that the model's performance selecting a subset of features
# decreases compared with the model using all available features. Since we
# generated the dataset, we can infer that the decrease is because the
# selection did not choose the two informative features.
#
# We can quickly investigate which feature have been selected during the
# cross-validation. We will print the indices of the two selected features.

# %%
import numpy as np

for idx, pipeline in enumerate(cv_results_with_selection["estimator"]):
    print(
        f"Fold #{idx} - features selected are: "
        f"{np.argsort(pipeline[0].scores_)[-2:]}"
    )

# %% [markdown]
# We see that the feature `53` is always selected while the other feature
# varies depending on the cross-validation fold.
#
# If we would like to keep our score with similar performance, we could choose
# another metric to perform the test or select more features. For instance, we
# could select the number of features based on a specific percentile of the
# highest scores. Besides, we should keep in mind that we simplify our problem
# by having informative and not informative features. Correlation between
# features makes the problem of feature selection even harder.
#
# Therefore, we could come with a much more complicated procedure that could
# fine-tune (via cross-validation) the number of selected features and change
# the way feature is selected (e.g. using a machine-learning model). However,
# going towards these solutions alienates the feature selection's primary
# purpose to get a significant train/test speed-up. Also, if the primary goal
# was to get a more performant model, performant models exclude non-informative
# features natively.
#
# ## Caveats of the feature selection
# When using feature selection, one has to be extra careful about the way it
# implements it. We will show two examples where feature selection can
# miserably fail.
#
# ### Selecting features without cross-validation
# TODO: find an example
#
# ### Limitation of selecting feature using a model
# An advanced strategy to select features is to use a machine learning model.
# Indeed, one can inspect a model and find relative feature importances. For
# instance, the parameters `coef_` for the linear models or
# `feature_importances_` for the tree-based models carries such information.
# Therefore, this method works as far as the relative feature importances given
# by the model is sufficient to select the meaningful feature.
#
# As presented in the model inspection notebook, the decision tree's relative
# feature importance will overestimate the importance of random feature when
# the decision tree overfits the training set.
#
# Here, we will generate a dataset that contains a large number of random
# features. Thus, when using a random forest to select features, we can expect
# that some of the random features will be wrongly selected.

# %%
X, y = make_classification(
    n_samples=5000,
    n_features=100,
    n_informative=2,
    n_redundant=5,
    n_repeated=5,
    class_sep=0.3,
    random_state=0,
)

# %% [markdown]
# First, let's build a model which will not make any features selection. We
# will use a cross-validation to evaluate this model.

# %%
model_without_selection = RandomForestClassifier(n_jobs=-1)
cv_results_without_selection = pd.DataFrame(
    cross_validate(model_without_selection, X, y, cv=5)
)

# %% [markdown]
# Then, we will build another model which will include a feature selection
# step based on a random forest. We will also evaluate the performance of the
# model via cross-validation.

# %%
from sklearn.feature_selection import SelectFromModel

model_with_selection = make_pipeline(
    SelectFromModel(
        estimator=RandomForestClassifier(n_jobs=-1),
    ),
    RandomForestClassifier(n_jobs=-1),
)
cv_results_with_selection = pd.DataFrame(
    cross_validate(model_with_selection, X, y, cv=5)
)

# %% [markdown]
# We can compare the generalization score of the two models.

# %%
cv_results = pd.concat(
    [cv_results_without_selection, cv_results_with_selection],
    axis=1,
    keys=["Without feature selection", "With feature selection"],
).swaplevel(axis="columns")
cv_results["test_score"].plot.box(vert=False, whis=100)
plt.xlabel("Accuracy")
_ = plt.title("Limitation of using a random forest for feature selection")

# %% [markdown]
# The model that selected a subset of feature is less performant than a
# random forest fitted on the full dataset. This is due to the fact that the
# random features have been used extensively by the random forest which is the
# case with overfit forest.
