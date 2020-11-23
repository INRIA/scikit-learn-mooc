# %% [markdown]
# # Speeding-up gradient-boosting
# In this notebook, we present a modified version of gradient boosting which
# uses a reduce number of split when building the different trees. This
# algorithm is called histogram gradient boosting in scikit-learn.
#
# We previously mentioned that random-forest is an efficient algorithm since
# each tree of the ensemble can be fitted at the same time independently.
# Therefore, the algorithm scales efficiently with both the number of CPUs and
# the number of samples.
#
# In gradient-boosting, the algorithm is a sequential algorithm. It requires
# the `N-1` trees to have been fit to be able to fit the tree at stage `N`.
# Therefore, the algorithm is quite computationally expensive. The most
# expensive part in this algorithm is the search for the best split in the tree
# which is a brute-force approach: all possible split are evaluated and the
# best one is picked. We explained this process in the notebook "tree in
# depth", which you can refer to.
#
# To accelerate the gradient-boosting algorithm, one could reduce the number of
# splits to be evaluated. As a consequence, the performance of such a
# tree would be reduced. However, since we are combining several trees in a
# gradient-boosting, we can add more estimators to overcome
# this issue.
#
# We will make a naive implementation of such algorithm using building blocks
# from scikit-learn. First, we will load the california housing dataset.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
# We will make a quick benchmark of the original gradient boosting.

# %%
from time import time
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting = GradientBoostingRegressor(n_estimators=200)

start_time = time()
gradient_boosting.fit(X_train, y_train)
fit_time_gradient_boosting = time() - start_time

start_time = time()
score_gradient_boosting = gradient_boosting.score(X_test, y_test)
score_time_gradient_boosting = time() - start_time

print("Gradient boosting decision tree")
print(f"R2 score: {score_gradient_boosting:.3f}")
print(f"Fit time: {fit_time_gradient_boosting:.2f} s")
print(f"Score time: {score_time_gradient_boosting:.5f} s\n")

# %% [markdown]
# We recall that a way to accelerate the gradient boosting is to reduce the
# number of split considered within the tree building. One way is to bin the
# data before to give them into the gradient boosting. A transformer called
# `KBinsDiscretizer` is doing such transformation. Thus, we can pipeline
# this preprocessing with the gradient boosting.
#
# We can first demonstrate the transformation done by the `KBinsDiscretizer`.

# %%
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(
    n_bins=256, encode="ordinal", strategy="quantile")
X_trans = discretizer.fit_transform(X_train)
X_trans

# %% [markdown]
# We see that the discretizer transform the original data into an integer.
# This integer represents the bin index when the distribution by quantile is
# performed. We can check the number of bins per feature.

# %%
[len(np.unique(col)) for col in X_trans.T]

# %% [markdown]
# After this transformation, we see that we have at most 256 unique values per
# features. Now, we will use this transformer to discretize data before to
# train the gradient boosting regressor.

# %%
from sklearn.pipeline import make_pipeline

gradient_boosting = make_pipeline(
    discretizer, GradientBoostingRegressor(n_estimators=200))

start_time = time()
gradient_boosting.fit(X_train, y_train)
fit_time_gradient_boosting = time() - start_time

start_time = time()
score_gradient_boosting = gradient_boosting.score(X_test, y_test)
score_time_gradient_boosting = time() - start_time

print("KBinsDiscritizer + Gradient boosting decision tree")
print(f"R2 score: {score_gradient_boosting:.3f}")
print(f"Fit time: {fit_time_gradient_boosting:.2f} s")
print(f"Score time: {score_time_gradient_boosting:.5f} s\n")

# %% [markdown]
# Here, we observe that the fit time have been drastically reduce but that the
# performance of the model are the identical. Scikit-learn provides a specific
# class even more optimized for large dataset called
# `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`. Each
# feature in the dataset `X` is first binned by computing histograms which are
# later used to evaluate the potential splits. The number of splits to evaluate
# is then much smaller. This algorithm becomes much more efficient than
# gradient boosting when the dataset has 10,000+ samples.
#
# Below we will give an example of a large dataset and we can compare
# computation time with the earlier experiment in the previous section.

# %%
from time import time
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

histogram_gradient_boosting = HistGradientBoostingRegressor(
    max_iter=200, random_state=0)

start_time = time()
histogram_gradient_boosting.fit(X_train, y_train)
fit_time_histogram_gradient_boosting = time() - start_time

start_time = time()
score_histogram_gradient_boosting = histogram_gradient_boosting.score(
    X_test, y_test)
score_time_histogram_gradient_boosting = time() - start_time

print("Historgram gradient boosting decision tree")
print(f"R2 score: {score_histogram_gradient_boosting:.3f}")
print(f"Fit time: {fit_time_histogram_gradient_boosting:.2f} s")
print(f"Score time: {score_time_histogram_gradient_boosting:.5f} s\n")

# %% [markdown]
# The histogram gradient-boosting is the best algorithm in terms of score.
# It will also scale when the number of samples increases, while the normal
# gradient-boosting will not.
