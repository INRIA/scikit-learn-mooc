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

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # Supervised Learning Part 2 -- Regression Analysis

# %% [markdown]
# In regression we are trying to predict a continuous output variable -- in contrast to the nominal variables we were predicting in the previous classification examples. 
#
# Let's start with a simple toy example with one feature dimension (explanatory variable) and one target variable. We will create a dataset out of a sine curve with some noise:

# %%
x = np.linspace(-3, 3, 100)
print(x)

# %%
rng = np.random.RandomState(42)
y = np.sin(4 * x) + x + rng.uniform(size=len(x))

# %%
plt.plot(x, y, 'o');

# %% [markdown]
# Linear Regression
# =================
#
# The first model that we will introduce is the so-called simple linear regression. Here, we want to fit a line to the data, which 
#
# One of the simplest models again is a linear one, that simply tries to predict the data as lying on a line. One way to find such a line is `LinearRegression` (also known as [*Ordinary Least Squares (OLS)*](https://en.wikipedia.org/wiki/Ordinary_least_squares) regression).
# The interface for LinearRegression is exactly the same as for the classifiers before, only that ``y`` now contains float values, instead of classes.

# %% [markdown]
# As we remember, the scikit-learn API requires us to provide the target variable (`y`) as a 1-dimensional array; scikit-learn's API expects the samples (`X`) in form a 2-dimensional array -- even though it may only consist of 1 feature. Thus, let us convert the 1-dimensional `x` NumPy array into an `X` array with 2 axes:
#

# %%
print('Before: ', x.shape)
X = x[:, np.newaxis]
print('After: ', X.shape)

# %% [markdown]
# Again, we start by splitting our dataset into a training (75%) and a test set (25%):

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %% [markdown]
# Next, we use the learning algorithm implemented in `LinearRegression` to **fit a regression model to the training data**:

# %%
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %% [markdown]
# After fitting to the training data, we paramerterized a linear regression model with the following values.

# %%
print('Weight coefficients: ', regressor.coef_)
print('y-axis intercept: ', regressor.intercept_)

# %% [markdown]
# Since our regression model is a linear one, the relationship between the target variable (y) and the feature variable (x) is defined as 
#
# $$y = \text{weight} \times x + \text{intercept .}$$
#
# Plugging in the min and max values into thos equation, we can plot the regression fit to our training data:

# %%
min_pt = X.min() * regressor.coef_[0] + regressor.intercept_
max_pt = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([X.min(), X.max()], [min_pt, max_pt])
plt.plot(X_train, y_train, 'o');

# %% [markdown]
# Similar to the estimators for classification in the previous notebook, we use the `predict` method to predict the target variable. And we expect these predicted values to fall onto the line that we plotted previously:

# %%
y_pred_train = regressor.predict(X_train)

# %%
plt.plot(X_train, y_train, 'o', label="data")
plt.plot(X_train, y_pred_train, 'o', label="prediction")
plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best')

# %% [markdown]
# As we can see in the plot above, the line is able to capture the general slope of the data, but not many details.

# %% [markdown]
# Next, let's try the test set:

# %%
y_pred_test = regressor.predict(X_test)

# %%
plt.plot(X_test, y_test, 'o', label="data")
plt.plot(X_test, y_pred_test, 'o', label="prediction")
plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best');

# %% [markdown]
# Again, scikit-learn provides an easy way to evaluate the prediction quantitatively using the ``score`` method. For regression tasks, this is the R<sup>2</sup> score. Another popular way would be the Mean Squared Error (MSE). As its name implies, the MSE is simply the average squared difference over the predicted and actual target values
#
# $$MSE = \frac{1}{n} \sum_{i=1}^{n} (\text{predicted}_i - \text{true}_i)^2$$

# %%
regressor.score(X_test, y_test)

# %% [markdown]
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Add a (non-linear) feature containing  `sin(4x)` to `X` and redo the fit as a new column to X_train (and X_test). Visualize the predictions with this new richer, yet linear, model.
#       </li>
#       <li>
#       Hint: you can use `np.concatenate(A, B, axis=1)` to concatenate two matrices A and B horizontal (to combine the columns).
#       </li>
#     </ul>
# </div>

# %%
# # %load solutions/06B_lin_with_sine.py

# %% [markdown]
# KNeighborsRegression
# =======================
# As for classification, we can also use a neighbor based method for regression. We can simply take the output of the nearest point, or we could average several nearest points. This method is less popular for regression than for classification, but still a good baseline.

# %%
from sklearn.neighbors import KNeighborsRegressor
kneighbor_regression = KNeighborsRegressor(n_neighbors=1)
kneighbor_regression.fit(X_train, y_train)

# %% [markdown]
# Again, let us look at the behavior on training and test set:

# %%
y_pred_train = kneighbor_regression.predict(X_train)

plt.plot(X_train, y_train, 'o', label="data", markersize=10)
plt.plot(X_train, y_pred_train, 's', label="prediction", markersize=4)
plt.legend(loc='best');

# %% [markdown]
# On the training set, we do a perfect job: each point is its own nearest neighbor!

# %%
y_pred_test = kneighbor_regression.predict(X_test)

plt.plot(X_test, y_test, 'o', label="data", markersize=8)
plt.plot(X_test, y_pred_test, 's', label="prediction", markersize=4)
plt.legend(loc='best');

# %% [markdown]
# On the test set, we also do a better job of capturing the variation, but our estimates look much messier than before.
# Let us look at the R<sup>2</sup> score:

# %%
kneighbor_regression.score(X_test, y_test)

# %% [markdown]
# Much better than before! Here, the linear model was not a good fit for our problem; it was lacking in complexity and thus under-fit our data.

# %% [markdown]
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Compare the KNeighborsRegressor and LinearRegression on the boston housing dataset. You can load the dataset using ``sklearn.datasets.load_boston``. You can learn about the dataset by reading the ``DESCR`` attribute.
#       </li>
#     </ul>
# </div>

# %%

# %%
# # %load solutions/06A_knn_vs_linreg.py
