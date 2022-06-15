# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M7.02
#
# We presented different classification metrics in the previous notebook.
# However, we did not use it with a cross-validation. This exercise aims at
# practicing and implementing cross-validation.
#
# We will reuse the blood transfusion dataset.

# %%
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %% [markdown]
# First, create a decision tree classifier.

# %%
# solution
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

# %% [markdown]
# Create a `StratifiedKFold` cross-validation object. Then use it inside the
# `cross_val_score` function to evaluate the decision tree. We will first use
# the accuracy as a score function. Explicitly use the `scoring` parameter
# of `cross_val_score` to compute the accuracy (even if this is the default
# score). Check its documentation to learn how to do that.

# %%
# solution
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=10)
scores = cross_val_score(tree, data, target, cv=cv, scoring="accuracy")
print(f"Accuracy score: {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# Repeat the experiment by computing the `balanced_accuracy`.

# %%
# solution
scores = cross_val_score(tree, data, target, cv=cv,
                         scoring="balanced_accuracy")
print(f"Balanced accuracy score: {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# We will now add a bit of complexity. We would like to compute the precision
# of our model. However, during the course we saw that we need to mention the
# positive label which in our case we consider to be the class `donated`.
#
# We will show that computing the precision without providing the positive
# label will not be supported by scikit-learn because it is indeed ambiguous.

# %%
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
try:
    scores = cross_val_score(tree, data, target, cv=10, scoring="precision")
except ValueError as exc:
    print(exc)

# %% [markdown]
# ```{tip}
# We catch the exception with a `try`/`except` pattern to be able to print it.
# ```
# We get an exception because the default scorer has its positive label set to
# one (`pos_label=1`), which is not our case (our positive label is "donated").
# In this case, we need to create a scorer using the scoring function and the
# helper function `make_scorer`.
#
# So, import `sklearn.metrics.make_scorer` and
# `sklearn.metrics.precision_score`. Check their documentations for more
# information.
# Finally, create a scorer by calling `make_scorer` using the score function
# `precision_score` and pass the extra parameter `pos_label="donated"`.

# %%
# solution
from sklearn.metrics import make_scorer, precision_score

precision = make_scorer(precision_score, pos_label="donated")

# %% [markdown]
# Now, instead of providing the string `"precision"` to the `scoring` parameter
# in the `cross_val_score` call, pass the scorer that you created above.

# %%
# solution
scores = cross_val_score(tree, data, target, cv=cv, scoring=precision)
print(f"Precision score: {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# `cross_val_score` will only compute a single score provided to the `scoring`
# parameter. The function `cross_validate` allows the computation of multiple
# scores by passing a list of string or scorer to the parameter `scoring`,
# which could be handy.
#
# Import `sklearn.model_selection.cross_validate` and compute the accuracy and
# balanced accuracy through cross-validation. Plot the cross-validation score
# for both metrics using a box plot.

# %%
# solution
from sklearn.model_selection import cross_validate
scoring = ["accuracy", "balanced_accuracy"]

scores = cross_validate(tree, data, target, cv=cv, scoring=scoring)
scores

# %% tags=["solution"]
import pandas as pd

color = {"whiskers": "black", "medians": "black", "caps": "black"}

metrics = pd.DataFrame(
    [scores["test_accuracy"], scores["test_balanced_accuracy"]],
    index=["Accuracy", "Balanced accuracy"]
).T

# %% tags=["solution"]
import matplotlib.pyplot as plt

metrics.plot.box(vert=False, color=color)
_ = plt.title("Computation of multiple scores using cross_validate")
