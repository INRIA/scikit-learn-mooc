# %% [markdown]
# # Stratification
# Let's start with the concept of stratification by giving an example where
# we can get into trouble if we are not careful. We load the iris dataset.

# %%
from sklearn.datasets import load_iris

X, y = load_iris(as_frame=True, return_X_y=True)

# %% [markdown]
# At this point, we create a basic machine-learning model: a logistic
# regression. We expect this model to work quite well on the iris dataset since
# this is a toy dataset.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())

# %% [markdown]
# Once we created our model, we will use the cross-validation framework to
# evaluate it. We will use a strategy called `KFold` cross-validation. We give
# a simple usage example of the `KFold` strategy to get an intuition of how it
# splits the dataset. We will define a dataset with nine samples and repeat the
# cross-validation three times (i.e. `n_splits`).

# %%
import numpy as np
from sklearn.model_selection import KFold

X_random = np.random.randn(9, 1)
cv = KFold(n_splits=3)
for train_index, test_index in cv.split(X_random):
    print("TRAIN:", train_index, "TEST:", test_index)

# %% [markdown]
# By defining three splits, we will use three samples each time for testing and
# six for training. `KFold` does not shuffle by default. It means that it will
# select the three first samples for the testing set at the first split, then
# the three next three samples for the second split, and the three next for the
# last split. In the end, all samples have been used in testing at least once
# among the different splits.
#
# Now, let's apply this strategy to check the performance of our model.

# %%
from sklearn.model_selection import cross_validate

cv = KFold(n_splits=3)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %% [markdown]
# It is a real surprise that our model cannot correctly classify any sample in
# any cross-validation split. We will now check our target's value to
# understand the issue while we should have started with this step.

# %%
import seaborn as sns
sns.set_context("talk")

ax = y.plot()
ax.set_xlabel("Sample index")
ax.set_ylabel("Class")
ax.set_yticks(y.unique())
_ = ax.set_title("Class value in target y")

# %% [markdown]
# We see that the target vector `y` is ordered. It will have some unexpected
# consequences when using the `KFold` cross-validation. To illustrate the
# consequences, we will show the class count in each fold of the
# cross-validation in the train and test set.
#
# For this matter, we create a function, because we will reuse it, which given
# a cross-validation object and the data `X` and `y`, is returning a dataframe
# with the class counts by folds and by split sets.

# %%
from collections import Counter
import pandas as pd


def compute_class_count_cv(cv, X, y):
    class_probability = []
    for cv_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        # Compute the class probability for the training set
        train_class = Counter(y[train_index])
        class_probability += [
            ["Train set", f"CV #{cv_idx}", klass, proportion]
            for klass, proportion in train_class.items()
        ]
        # Compute the class probability for the test set
        test_class = Counter(y[test_index])
        class_probability += [
            ["Test set", f"CV #{cv_idx}", klass, proportion]
            for klass, proportion in test_class.items()
        ]

    class_probability = pd.DataFrame(
        class_probability, columns=["Set", "CV", "Class", "Count"])
    return class_probability


# %% [markdown]
# Let's compute the statistics using the `KFold` cross-validation and we will
# plot these information in a bar plot.

# %%
kfold_class_count = compute_class_count_cv(cv, X, y)
kfold_class_count

# %%
g = sns.FacetGrid(kfold_class_count, col="Set")
g.map_dataframe(
    sns.barplot, x="Class", y="Count", hue="CV", palette="tab10")
g.set_axis_labels("Class", "Count")
g.add_legend()
_ = g.fig.suptitle("Class count with K-fold cross-validation", y=1.05)

# %% [markdown]
# We can confirm that in each fold, only two of the three classes are present
# in the training set and all samples of the remaining class is used as a test
# set. So our model is unable to predict this class that was unseen during the
# training stage.
#
# One possibility to solve the issue is to shuffle the data before to split the
# data into three groups.

# %%
cv = KFold(n_splits=3, shuffle=True, random_state=0)
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %% [markdown]
# We get results that are closer to what we would expect with an accuracy above
# 90%. Now that we solved our first issue, it would be interesting to check if
# the class frequency in the training and testing set is equal to our original
# set's class frequency. It would ensure that we are training and testing our
# model with a class distribution that we will encounter in production.

# %%
kfold_shuffled_class_count = compute_class_count_cv(cv, X, y)

g = sns.FacetGrid(kfold_shuffled_class_count, col="Set")
g.map_dataframe(
    sns.barplot, x="Class", y="Count", hue="CV", palette="tab10")
g.set_axis_labels("Class", "Count")
g.add_legend()
_ = g.fig.suptitle(
    "Class count with shuffled K-fold cross-validation", y=1.05)

# %% [markdown]
# We see that neither the training and testing sets have the same class
# frequencies as our original dataset because the count for each class is
# varying a little.
#
# However, one might want to split our data by preserving the original class
# frequencies: we want to **stratify** our data by class. In scikit-learn, some
# cross-validation strategies are implementing the stratification and contains
# `Stratified` in their names.

# %%
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)

# %%
results = cross_validate(model, X, y, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is "
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %%
stratified_kfold_class_count = compute_class_count_cv(cv, X, y)

g = sns.FacetGrid(stratified_kfold_class_count, col="Set")
g.map_dataframe(
    sns.barplot, x="Class", y="Count", hue="CV", palette="tab10")
g.set_axis_labels("Class", "Count")
g.add_legend()
_ = g.fig.suptitle(
    "Class count with stratifiedK-fold cross-validation", y=1.05)

# %% [markdown]
# In this case, we observe that the class counts either in the train set or the
# test set are very close. The difference is due to the small number of samples
# in the iris dataset.
#
# In conclusion, this is a good practice to use stratification within the
# cross-validation framework when dealing with a classification problem.
