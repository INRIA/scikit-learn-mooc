# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"deletable": true, "editable": true, "hide_input": false}
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown] {"deletable": true, "editable": true}
# ## Automatic Feature Selection
# Often we collected many features that might be related to a supervised prediction task, but we don't know which of them are actually predictive. To improve interpretability, and sometimes also generalization performance, we can use automatic feature selection to select a subset of the original features. There are several types of feature selection methods available, which we'll explain in order of increasing complexity.
#
# For a given supervised model, the best feature selection strategy would be to try out each possible subset of the features, and evaluate generalization performance using this subset. However, there are exponentially many subsets of features, so this exhaustive search is generally infeasible. The strategies discussed below can be thought of as proxies for this infeasible computation.
#
# ### Univariate statistics
# The simplest method to select features is using univariate statistics, that is by looking at each feature individually and running a statistical test to see whether it is related to the target. This kind of test is also known as analysis of variance (ANOVA).
#
# We create a synthetic dataset that consists of the breast cancer data with an additional 50 completely random features.

# %% {"deletable": true, "editable": true}
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target,
                                                    random_state=0, test_size=.5)

# %% [markdown] {"deletable": true, "editable": true}
# We have to define a threshold on the p-value of the statistical test to decide how many features to keep. There are several strategies implemented in scikit-learn, a straight-forward one being ``SelectPercentile``, which selects a percentile of the original features (we select 50% below):

# %% {"deletable": true, "editable": true}
from sklearn.feature_selection import SelectPercentile

# use f_classif (the default) and SelectPercentile to select 50% of features:
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set:
X_train_selected = select.transform(X_train)

print(X_train.shape)
print(X_train_selected.shape)

# %% [markdown] {"deletable": true, "editable": true}
# We can also use the test statistic directly to see how relevant each feature is. As the breast cancer dataset is a classification task, we use f_classif, the F-test for classification. Below we plot the p-values associated with each of the 80 features (30 original features + 50 noise features). Low p-values indicate informative features.

# %% {"deletable": true, "editable": true}
from sklearn.feature_selection import f_classif, f_regression, chi2

# %% {"deletable": true, "editable": true}
F, p = f_classif(X_train, y_train)

# %% {"deletable": true, "editable": true}
plt.figure()
plt.plot(p, 'o')

# %% [markdown] {"deletable": true, "editable": true}
# Clearly most of the first 30 features have very small p-values.
#
# Going back to the SelectPercentile transformer, we can obtain the features that are selected using the ``get_support`` method:

# %% {"deletable": true, "editable": true}
mask = select.get_support()
print(mask)
# visualize the mask. black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')

# %% [markdown] {"deletable": true, "editable": true}
# Nearly all of the original 30 features were recovered.
# We can also analize the utility of the feature selection by training a supervised model on the data.
# It's important to learn the feature selection only on the training set!

# %% {"deletable": true, "editable": true}
from sklearn.linear_model import LogisticRegression

# transform test data:
X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: %f" % lr.score(X_test, y_test))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: %f" % lr.score(X_test_selected, y_test))

# %% [markdown] {"deletable": true, "editable": true}
# ### Model-based Feature Selection
# A somewhat more sophisticated method for feature selection is using a supervised machine learning model and selecting features based on how important they were deemed by the model. This requires the model to provide some way to rank the features by importance. This can be done for all tree-based models (which implement ``get_feature_importances``) and all linear models, for which the coefficients can be used to determine how much influence a feature has on the outcome.
#
# Any of these models can be made into a transformer that does feature selection by wrapping it with the ``SelectFromModel`` class:

# %% {"deletable": true, "editable": true}
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")

# %% {"deletable": true, "editable": true}
select.fit(X_train, y_train)
X_train_rf = select.transform(X_train)
print(X_train.shape)
print(X_train_rf.shape)

# %% {"deletable": true, "editable": true}
mask = select.get_support()
# visualize the mask. black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')

# %% {"deletable": true, "editable": true}
X_test_rf = select.transform(X_test)
LogisticRegression().fit(X_train_rf, y_train).score(X_test_rf, y_test)

# %% [markdown] {"deletable": true, "editable": true}
# This method builds a single model (in this case a random forest) and uses the feature importances from this model.
# We can do a somewhat more elaborate search by training multiple models on subsets of the data. One particular strategy is recursive feature elimination:

# %% [markdown] {"deletable": true, "editable": true}
# ### Recursive Feature Elimination
# Recursive feature elimination builds a model on the full set of features, and similar to the method above selects a subset of features that are deemed most important by the model. However, usually only a single feature is dropped from the dataset, and a new model is built with the remaining features. The process of dropping features and model building is repeated until there are only a pre-specified number of features left:

# %% {"deletable": true, "editable": true}
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)

select.fit(X_train, y_train)
# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')

# %% {"deletable": true, "editable": true}
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)

# %% {"deletable": true, "editable": true}
select.score(X_test, y_test)

# %% [markdown] {"deletable": true, "editable": true}
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Create the "XOR" dataset as in the first cell below:
#       </li>
#       <li>
#       Add random features to it and compare how univariate selection compares to model based selection using a Random Forest in recovering the original features.
#       </li>
#     </ul>
# </div>

# %% {"deletable": true, "editable": true}
import numpy as np

rng = np.random.RandomState(1)

# Generate 400 random integers in the range [0, 1]
X = rng.randint(0, 2, (200, 2))
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)  # XOR creation

plt.scatter(X[:, 0], X[:, 1], c=plt.cm.tab10(y))

# %% {"deletable": true, "editable": true}
# # %load solutions/19_univariate_vs_mb_selection.py
