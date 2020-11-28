# %% [markdown]
# # Solution for Exercise 03
#
# The goal of this exercise is to find the limitation blindly a k-fold
# cross-validation.
#
# We will use the iris dataset to demonstrate one of the issue.

# %%
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)

# %% [markdown]
# Create a decision tree classifier that we will use in the next experiments.

# %%
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# %% [markdown]
# As a first experiment, use the utility
# `sklearn.model_selection.train_test_split` to split the data into a train
# and test set. Train the classifier using the train set and check the score
# on the test set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

model.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# We get good results which is not surprising because this classification
# is a an easy problem.
#
# Now, use the utility `sklearn.utils.cross_val_score` with a
# `sklearn.model_selection.KFold` by setting only `n_splits=3`. Check the
# results on each fold. Explain the results.

# %%
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=3)
cross_val_score(model, X, y, cv=cv)

# %% [markdown]
# We observe that we have an accuracy of 0 on each fold. We will see more
# in details in the lecture the reason but in short, the target `y` was ordered
# which cause the issue.
