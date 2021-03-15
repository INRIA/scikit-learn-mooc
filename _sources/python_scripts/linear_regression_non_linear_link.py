# %% [markdown]
# # Linear regression with non-linear link between data and target
#
# In the previous exercise, you were asked to train a linear regression model
# on a dataset where the matrix `data` and the vector `target` do not have a
# linear link.
#
# In this notebook, we show that even if the parametrization of linear models
# is not natively adapated to data with non-linearity, it is still possible
# to make linear model more flexible and expressive.
#
# To illustrate these concepts, we will reuse the same dataset generated in the
# previous exercise.

# %%
import numpy as np

rng = np.random.RandomState(0)

n_sample = 100
data_max, data_min = 1.4, -1.4
len_data = (data_max - data_min)
# sort the data to make plotting easier later
data = np.sort(rng.rand(n_sample) * len_data - len_data / 2)
noise = rng.randn(n_sample) * .3
target = data ** 3 - 0.5 * data ** 2 + noise

# %% [markdown]
# ```{note}
# To ease the plotting, we will create a Pandas dataframe containing the data
# and target
# ```

# %%
import pandas as pd
full_data = pd.DataFrame({"data": data, "target": target})

# %%
import seaborn as sns

_ = sns.scatterplot(data=full_data, x="data", y="target")
# %% [markdown]
# We will highlight the limitations of fitting a linear regression model as
# done in the previous exercise.
#
# ```{warning}
# In scikit-learn, by convention `data` (also called `X` in the scikit-learn
# documentation) should be a 2D matrix of shape `(n_samples, n_features)`.
# If `data` is a 1D vector, you need to reshape it into a matrix with a
# single column if the vector represents a feature or a single row if the
# vector represents a sample.
# ```

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_regression = LinearRegression()
# X should be 2D for sklearn
data_2d = data.reshape((-1, 1))
linear_regression.fit(data_2d, target)
target_predicted = linear_regression.predict(data_2d)

# %%
mse = mean_squared_error(target, target_predicted)

# %%
ax = sns.scatterplot(data=full_data, x="data", y="target")
ax.plot(data, target_predicted, color="tab:orange")
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# Here the coefficients learnt by `LinearRegression` is the best "straight
# line" that fits the data. We can inspect the coefficients using the
# attributes of the model learnt as follows:

# %%
print(f"weight: {linear_regression.coef_[0]:.2f}, "
      f"intercept: {linear_regression.intercept_:.2f}")

# %% [markdown]
# It is important to note that the model learnt will not be able to handle the
# non-linear relationship between `data` and `target` since linear models
# assume the relationship between `data` and `target` to be linear.
#
# Indeed, there are 3 possibilities to solve this issue:
#
# 1. choose a model that natively can deal with non-linearity,
# 2. "augment" features by including expert knowledge which can be used by
#    the model, or
# 3. use a "kernel" to have a locally-based decision function instead of a
#    global linear decision function.
#
# Let's illustrate quickly the first point by using a decision tree regressor
# which can natively handle non-linearity.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3).fit(data_2d, target)
target_predicted = tree.predict(data_2d)
mse = mean_squared_error(target, target_predicted)

# %%
ax = sns.scatterplot(data=full_data, x="data", y="target")
ax.plot(data, target_predicted, color="tab:orange")
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# Instead of having a model which can natively deal with non-linearity, we
# could also modify our data: we could create new features, derived from the
# original features, using some expert knowledge. In this example, we know that
# we have a cubic and squared relationship between `data` and `target` (because
# we generated the data). Indeed, we could create two new features (`data ** 2`
# and `data ** 3`) using this information.

# %%
data_augmented = np.concatenate([data_2d, data_2d ** 2, data_2d ** 3], axis=1)

linear_regression.fit(data_augmented, target)
target_predicted = linear_regression.predict(data_augmented)
mse = mean_squared_error(target, target_predicted)

# %%
ax = sns.scatterplot(data=full_data, x="data", y="target")
ax.plot(data, target_predicted, color="tab:orange")
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# We can see that even with a linear model, we can overcome the linearity
# limitation of the model by adding the non-linear component into the design of
# additional features. Here, we created new features by knowing the way the
# target was generated. In practice, this is usually not the case.
#
# Instead, one is usually creating interaction between features (e.g. $x_1
# \times x_2$) with different orders (e.g. $x_1, x_1^2, x_1^3$), at the risk of
# creating a model with too much flexibility where the polynomial terms allows
# to fit noise in the dataset and thus lead overfit. In scikit-learn, the
# `PolynomialFeatures` is a transformer to create such feature interactions
# which we could have used instead of manually creating new features.
#
# To demonstrate `PolynomialFeatures`, we are going to use a scikit-learn
# pipeline which will first create the new features and then fit the model.
# We come back to scikit-learn pipelines and discuss them in more detail later.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

model = make_pipeline(PolynomialFeatures(degree=3),
                      LinearRegression())
model.fit(data_2d, target)
target_predicted = model.predict(data_2d)
mse = mean_squared_error(target, target_predicted)

# %%
ax = sns.scatterplot(data=full_data, x="data", y="target")
ax.plot(data, target_predicted, color="tab:orange")
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# Thus, we saw that `PolynomialFeatures` is actually doing the same
# operation that we did manually above.
#
# The last possibility is to make a linear model more expressive is to use a
# "kernel". Instead of learning a weight per feature as we previously
# emphasized, a weight will be assign by sample instead. However, not all
# samples will be used. This is the base of the support vector machine
# algorithm.

# %%
from sklearn.svm import SVR

svr = SVR(kernel="linear")
svr.fit(data_2d, target)
target_predicted = svr.predict(data_2d)
mse = mean_squared_error(target, target_predicted)

# %%
ax = sns.scatterplot(data=full_data, x="data", y="target")
ax.plot(data, target_predicted, color="tab:orange")
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# The algorithm can be modified so that it can use non-linear kernel. Then,
# it will compute interaction between samples using this non-linear
# interaction.

# %%
svr = SVR(kernel="poly", degree=3)
svr.fit(data_2d, target)
target_predicted = svr.predict(data_2d)
mse = mean_squared_error(target, target_predicted)

# %%
ax = sns.scatterplot(data=full_data, x="data", y="target")
ax.plot(data, target_predicted, color="tab:orange")
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# A method supporting kernel, as SVM, allows to efficiently create a non-linear
# model.
