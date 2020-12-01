# %% [markdown]
# # 📃 Solution for Exercise 01
#
# We presented different classification metrics in the previous notebook.
# However, we did not use it with a cross-validation. This exercise aims at
# practising and implementing cross-validation that will use such score.
#
# We will reuse the same blood transfusion dataset.

# %%
import pandas as pd

data = pd.read_csv("../datasets/blood_transfusion.csv")
X, y = data.drop(columns="Class"), data["Class"]

# %% [markdown]
# First, create a decision tree classifier.

# %%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

# %% [markdown]
# Create a `StratifiedKFold` cross-validation object. Then use it inside the
# `cross_val_score` function to evaluate the decision tree. We will first use
# the accuracy as a score function. Check the documentation of the parameter
# `scoring` in `cross_val_score` to check how to explicitely mention to compute
# the accuracy (even if this is the default score).

# %%
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=10)
scores = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
print(f"Accuracy score: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# Repeat the experiment by computing the `balanced_accuracy`.

# %%
scores = cross_val_score(tree, X, y, cv=cv, scoring="balanced_accuracy")
print(f"Balanced accuracy score: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# We will now add a bit of complexity. We would like to compute the precision
# of our model. However, during the course we saw that we need to mention the
# positive label which in our case we consider to be the class `donated`.
#
# We will show that computing the precision without providing the positive
# label will not be supported by scikit-learn because it is indeed ambiguous.

# %%
try:
    scores = cross_val_score(tree, X, y, cv=cv, scoring="precision")
except ValueError as exc:
    print(exc)

# %% [markdown]
# We get an exception because by default `pos_label=1` which is not the case
# in our case. Thus, we would like to specify a parameter linked to the score
# function. In this case, we need to create a scorer using the scoring function
# and the helper function `make_scorer`.
#
# So, import `sklearn.metrics.make_scorer` and
# `sklearn.metrics.precision_score`. Check the documentation of `make_scorer`.
# Finally, create a scorer by calling `make_scorer` using the score function
# `precision_score` and pass the extra parameter `pos_label="donated"`.

# %%
from sklearn.metrics import make_scorer, precision_score

precision = make_scorer(precision_score, pos_label="donated")

# %% [markdown]
# Now, instead to provide the string `"precision"` to the `scoring` parameter
# in the `cross_val_score` call, pass the scorer that you created above.

# %%
scores = cross_val_score(tree, X, y, cv=cv, scoring=precision)
print(f"Precision score: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# `cross_val_score` will only compute a single score provided to the `scoring`
# parameter. The function `cross_validate` allows to compute multiple score
# which could be handy by passing a list of string or scorer to the parameter
# scoring.
#
# Import `sklearn.model_selection.cross_validate` and compute the accuracy and
# balanced accuracy through cross-validation. Plot the cross-validation score
# for both metrics using a box plot.

# %%
from sklearn.model_selection import cross_validate
scoring = ["accuracy", "balanced_accuracy"]

scores = cross_validate(tree, X, y, cv=cv, scoring=scoring)
scores

# %%
import pandas as pd
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

metrics = pd.DataFrame(
    [scores["test_accuracy"], scores["test_balanced_accuracy"]],
    index=["Accuracy", "Balanced accuracy"]
).T
ax = metrics.plot.box(**boxplot_property)
_ = ax.set_title("Computation of multiple scores using cross_validate")

# %%
