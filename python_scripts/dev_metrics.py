# %% [markdown]
# # Evaluate of your predictive model

# %% [markdown]
# ## Introduction
# Machine-learning models rely on optimizing an objective function, by seeking
# its minimum or maximum. It is important to understand that this objective
# function is usually decoupled from the evaluation metric that we want to
# optimize in practice. The objective function serves as a proxy to the
# evaluation metric.
#
# While other notebooks will give insights regarding algorithms and their
# associated objective functions, in this notebook we will focus on the
# metrics used to evaluate the performance of a predictive model.
#
# Selecting an evaluation metric will mainly depend on the model chosen to
# solve our datascience problem.

# %% [markdown]
# ## Classification
# We can recall that in a classification setting, the target `y` is categorical
# rather than continuous. We will use the blood transfusion dataset that will
# be fetched from OpenML.
import pandas as pd
from sklearn.datasets import fetch_openml

X, y = fetch_openml(
    name="blood-transfusion-service-center",
    as_frame=True, return_X_y=True,
)
# Make columns and classes more human-readable
X.columns = ["Recency", "Frequency", "Monetary", "Time"]
y = y.apply(
    lambda x: "donated" if x == "2" else "not donated"
).astype("category")
y.cat.categories
# %% [markdown]
# We can see that the target `y` contains 2 categories corresponding to whether
# or not a subject gave blood or not. We will use a logistic regression
# classifier to predict this outcome.
#
# First, we split the data into a training and a testing set.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0, test_size=0.5
)
# %% [markdown]
# Once our data are split, we can learn a logistic regression classifier solely
# on the training data, keeping the testing data for the evaluation of the
# model.
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# %% [markdown] Now, that our classifier is trained, we can provide a some
# information about a subject and the classifier can predict whether or not the
# subject will donate blood.
#
# Let's create a synthetic sample corresponding to the following potential
# donor: he/she donated blood 6 month ago and gave twice blood in the past for
# a total of 1000 c.c. He/she gave blood for the first time 20 months ago.
new_donor = [[6, 2, 1000, 20]]
classifier.predict(new_donor)
# %% [markdown]
# With these information, our classifier predicted that this synthetic subject
# is more likely to not donate blood. However, we have no possibility to ensure
# if the prediction is correct or not. That's why, we can now use the testing
# set for this purpose. First, we can predict whether or not a subject will
# give blood with the help of the trained classifier.
y_pred = classifier.predict(X_test)
y_pred
# %% [markdown]
# Now that we have these predictions, we could compare them with the true
# predictions (sometimes called ground-truth) which we did not use up to now.
y_test == y_pred
# %%[markdown]
# In the comparison above, a `True` value means that the value predicted by our
# classifier is identical to the real `prediction` while a `False` means that
# our classifier made a mistake. One way to get an overall statistic telling us
# how good the performance of our classifier are is the computer the number of
# time our classifier was right and divide it by the number of samples in our
# set.
import numpy as np

np.mean(y_test == y_pred)
# %% [markdown]
# This measure is also known as the accuracy. Here, our classifier is 97%
# accurate at classifying iris flowers. `scikit-learn` provides a function to
# compute this metric in the module `sklearn.metrics`.
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
# %% [markdown]
# Our classifier give the good answer in 77% of the case.
#
# Scikit-learn also have a build-in method named `score` which compute by
# default the accuracy score.
classifier.score(X_test, y_test)
# %% [markdown]
# The comparison that we did above and the accuracy that we deducted did not
# take into account which type of error our classifier was doing. The accuracy
# is an aggregate of the error. However, we might be interested in a lower
# granularity level to know separately the error for the two following case:
# - we predicted that a person will give blood but she/he is not;
# - we predicted that a person will not give blood but she/he is.
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier, X_test, y_test)
# %% [markdown]
# The in-diagonal numbers are related to predictions that agree with the true
# labels while off-diagonal numbers are related to misclassification. However,
# now we now exactly the sort of error made by our classifier.
# %% [markdown]
# ## Regression

# %% [markdown]
# ## Clustering
