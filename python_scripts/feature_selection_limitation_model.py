# %% [markdown]
# # Limitation of selecting feature using a model
# An advanced strategy to select features is to use a machine learning model.
# Indeed, one can inspect a model and find relative feature importances. For
# instance, the parameters `coef_` for the linear models or
# `feature_importances_` for the tree-based models carries such information.
# Therefore, this method works as far as the relative feature importances given
# by the model is sufficient to select the meaningful feature.
#
# Here, we will generate a dataset that contains a large number of random
# features.

# %%
from sklearn.datasets import make_classification

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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

model_without_selection = RandomForestClassifier(n_jobs=-1)
cv_results_without_selection = pd.DataFrame(
    cross_validate(model_without_selection, X, y, cv=5)
)

# %% [markdown]
# Then, we will build another model which will include a feature selection
# step based on a random forest. We will also evaluate the performance of the
# model via cross-validation.

# %%
from sklearn.pipeline import make_pipeline
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


# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

# Define the style of the box style
boxplot_property = {
    "vert": False, "whis": 100, "patch_artist": True, "widths": 0.3,
    "boxprops": dict(linewidth=3, color='black', alpha=0.9),
    "medianprops": dict(linewidth=2.5, color='black', alpha=0.9),
    "whiskerprops": dict(linewidth=3, color='black', alpha=0.9),
    "capprops": dict(linewidth=3, color='black', alpha=0.9),
}

cv_results["test_score"].plot.box(**boxplot_property)
plt.xlabel("Accuracy")
_ = plt.title("Limitation of using a random forest for feature selection")

# %% [markdown]
# The model that selected a subset of feature is less performant than a
# random forest fitted on the full dataset.
#
# We can rely on some aspects tackled in the notebook presenting the model
# inspection to explain this behaviour. The decision tree's relative feature
# importance will overestimate the importance of random feature when the
# decision tree overfits the training set.
#
# Therefore, it is good to keep in mind that feature selection relies on
# procedures making some assumptions, which can be perfectible.
