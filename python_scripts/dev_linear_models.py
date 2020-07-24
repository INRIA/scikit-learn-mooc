# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Models
#
# In this notebook we will review linear models from `scikit-learn`.
# We will :
# - learn how to fit a simple linear slope and interpret the coefficients;
# - discuss feature augmentation to fit a non-linear function;
# - use `LinearRegression` and its regularized version `Ridge` which is more
#   robust;
# - use `LogisticRegression` with `pipeline`;
# - see examples of linear separability.

# %% [markdown]
# ## 1. Regression

# %% [markdown]
# ### The over-simplistic toy example
# To illustrate the main principle of linear regression, we will use a dataset
# containing information about penguins.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins.csv")
data.head()

# %% [markdown]
# This dataset contains features of penguins. We will formulate the following
# problem. Observing the flipper length of a penguin, we would like to infer
# is mass.

# %%
import seaborn as sns

feature_names = "Flipper Length (mm)"
target_name = "Body Mass (g)"

sns.scatterplot(data=data, x=feature_names, y=target_name)

# select the features of interest
X = data[[feature_names]].dropna()
y = data[target_name].dropna()

# %% [markdown]
# In this problem, the mass of a penguin is our target. It is a continuous
# variable that roughly vary between 2700 g and 6300 g. Thus, this is a
# regression problem (in contrast to classification). We also see that there is
# almost linear relationship between the body mass of the penguin and the
# flipper length. Longer is the flipper, heavier is the penguin.
#
# Thus, we could come with a simple rule that given a length of the flipper
# we could compute the body mass of a penguin using a linear relationship of
# of the form `y = a * x + b` where `a` and `b` are the 2 parameters of our
# model.


# %%
def linear_model_flipper_mass(
    flipper_length, weight_flipper_length, intercept_body_mass
):
    """Linear model of the form y = a * x + b"""
    body_mass = weight_flipper_length * flipper_length + intercept_body_mass
    return body_mass


# %% [markdown]
# Using the model that we define, we can check which body mass values we would
# predict for a large range of flipper length.

# %%
import matplotlib.pyplot as plt
import numpy as np


def plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass,
    ax=None,
):
    """Compute and plot the prediction."""
    inferred_body_mass = linear_model_flipper_mass(
        flipper_length_range,
        weight_flipper_length=weight_flipper_length,
        intercept_body_mass=intercept_body_mass,
    )

    if ax is None:
        _, ax = plt.subplots()

    sns.scatterplot(data=data, x=feature_names, y=target_name, ax=ax)
    ax.plot(
        flipper_length_range,
        inferred_body_mass,
        linewidth=3,
        label=(
            f"{weight_flipper_length:.2f} (g / mm) * flipper length + "
            f"{intercept_body_mass:.2f} (g)"
        ),
    )
    plt.legend()


weight_flipper_length = 45
intercept_body_mass = -5000

flipper_length_range = np.linspace(X.min(), X.max(), num=300)
plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass
)

# %% [markdown]
# The variable `weight_flipper_length` is a weight applied to the feature in
# order to make the inference. When this coefficient is positive, it means that
# an increase of the flipper length will induce an increase of the body mass.
# If the coefficient is negative, an increase of the flipper length will induce
# a decrease of the body mass. Graphically, this coefficient is represented by
# the slope of the curve that we draw.

# %%
weight_flipper_length = -40
intercept_body_mass = 13000

flipper_length_range = np.linspace(X.min(), X.max(), num=300)
plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass
)


# %% [markdown]
# In our case, this coefficient has some meaningful unit. Indeed, its unit is
# g/mm. For instance, with a coefficient of 40 g/mm, it means that for an
# additional millimeter, the body weight predicted will increase of 40 g.

body_mass_180 = linear_model_flipper_mass(
    flipper_length=180, weight_flipper_length=40, intercept_body_mass=0
)
body_mass_181 = linear_model_flipper_mass(
    flipper_length=181, weight_flipper_length=40, intercept_body_mass=0
)

print(
    f"The body mass for a flipper length of 180 mm is {body_mass_180} g and "
    f"{body_mass_181} g for a flipper length of 181 mm"
)

# %% [markdown]
# We can also see that we have a parameter `intercept_body_mass` in our model.
# this parameter is the intercept of the curve when `x=0`. If the intercept is
# null, then the curve will be passing by the origin:

# %%
weight_flipper_length = 25
intercept_body_mass = 0

flipper_length_range = np.linspace(0, X.max(), num=300)
plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass
)

# %% [markdown]
# Otherwise, it will be the value intercepted:

# %%
weight_flipper_length = 45
intercept_body_mass = -5000

flipper_length_range = np.linspace(0, X.max(), num=300)
plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass
)

# %% [markdown]
# Now, that we understood how our model is inferring data, one should question
# on how to find the best value for the parameters. Indeed, it seems that we
# can have several model which will depend of the choice of parameters:

# %%
_, ax = plt.subplots()
flipper_length_range = np.linspace(X.min(), X.max(), num=300)
for weight, intercept in zip([-40, 45, 90], [15000, -5000, -14000]):
    plot_data_and_model(
        flipper_length_range, weight, intercept, ax=ax,
    )

# %% [markdown]
# To choose a model, we could use a metric indicating how good our model is
# fitting our data.

# %%
from sklearn.metrics import mean_squared_error

for weight, intercept in zip([-40, 45, 90], [15000, -5000, -14000]):
    inferred_body_mass = linear_model_flipper_mass(
        X,
        weight_flipper_length=weight,
        intercept_body_mass=intercept,
    )
    model_error = mean_squared_error(y, inferred_body_mass)
    print(
        f"The following model \n "
        f"{weight:.2f} (g / mm) * flipper length + {intercept:.2f} (g) \n"
        f"has a mean squared error of: {model_error:.2f}"
    )

# %% [markdown]
# Hopefully, this problem can be solved without the need to check every
# potential parameters combinations. Indeed, this problem as a closed-form
# solution (i.e. an equation giving the parameter values) avoiding for any
# brute-force search. This strategy is implemented in scikit-learn.

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X, y)

# %% [markdown]
# The instance `linear_regression` will store the parameter values in the
# attribute `coef_` and `intercept_`. We can check which is the optimal model
# found:

# %%
weight_flipper_length = linear_regression.coef_[0]
intercept_body_mass = linear_regression.intercept_

flipper_length_range = np.linspace(X.min(), X.max(), num=300)
plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass
)

inferred_body_mass = linear_regression.predict(X)
model_error = mean_squared_error(y, inferred_body_mass)
print(f"The error of the optimal model is {model_error:.2f}")

# %% [markdown]
# ### What if your data don't have a linear relationship
# Now, we will define a new problem where the feature and the target are not
# linearly linked. For instance, we could defined `x` to be the years of
# experience (normalized) and `y` the salary (normalized). Therefore, the
# problem here would be to infer the salary given the years of experience of
# someone.

# %%

# data generation
# fix the seed for reproduction
rng = np.random.RandomState(0)

n_sample = 100
x_max, x_min = 1.4, -1.4
len_x = (x_max - x_min)
x = rng.rand(n_sample) * len_x - len_x/2
noise = rng.randn(n_sample) * .3
y = x ** 3 - 0.5 * x ** 2 + noise

# plot the data
plt.scatter(x, y,  color='k', s=9)
plt.xlabel('x', size=26)
_ = plt.ylabel('y', size=26)

# %% [markdown]
# ### Exercise 1
#
# In this exercise, you are asked to approximate the target `y` by a linear
# function `f(x)`. i.e. find the best coefficients of the function `f` in order
# to minimize the error.
#
# Then you could compare the mean squared error of your model with the mean
# squared error of a linear model (which shall be the minimal one).


# %%
def f(x):
    w0 = 0  # TODO: update the weight here
    w1 = 0  # TODO: update the weight here
    y_predict = w1 * x + w0
    return y_predict


# plot the slope of f
grid = np.linspace(x_min, x_max, 300)
plt.scatter(x, y, color='k', s=9)
plt.plot(grid, f(grid), linewidth=3)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, f(x))
print(f"Mean squared error = {mse:.2f}")

# %% [markdown]
# ### Solution 1. by fiting a linear regression

# %%
from sklearn import linear_model

linear_regression = linear_model.LinearRegression()
# X should be 2D for sklearn
X = x.reshape((-1, 1))
linear_regression.fit(X, y)

# plot the best slope
y_best = linear_regression.predict(grid.reshape(-1, 1))
plt.plot(grid, y_best, linewidth=3)
plt.scatter(x, y, color="k", s=9)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, linear_regression.predict(X))
print(f"Lowest mean squared error = {mse:.2f}")

# %% [markdown]
# Here the coefficients learnt by `LinearRegression` is the best slope which
# fit the data. We can inspect those coefficients using the attributes of the
# model learnt as follow:

# %%
print(
    f"best coef: w1 = {linear_regression.coef_[0]:.2f}, "
    f"best intercept: w0 = {linear_regression.intercept_:.2f}"
)

# %% [markdown]
# It is important to note that the model learnt will not be able to handle
# the non-linearity linking `x` and `y` since it is beyond the assumption made
# when using a linear model. To obtain a better model, we have mainly 3
# solutions: (i) choose a model that natively can deal with non-linearity,
# (ii) "augment" features by including expert knowledge which can be used by
# the model, or (iii) use a "kernel" to have a locally-based decision function
# instead of a global linear decision function.
#
# Let's illustrate quickly the first point by using a decision tree regressor
# which natively can handle non-linearity.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3).fit(X, y)
y_pred = tree.predict(grid.reshape(-1, 1))

plt.plot(grid, y_pred, linewidth=3)
plt.scatter(x, y, color="k", s=9)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, tree.predict(X))
print(f"Lowest mean squared error = {mse:.2f}")

# %% [markdown]
# In this case, the model can handle the non-linearity. Instead having a model
# which natively can deal with non-linearity, we could modify our data: we
# could create new features, derived from the original features, using some
# expert knowledge. For instance, here we know that we have a cubic and squared
# relationship between `x` and `y` (because we generated the data). Indeed,
# we could create two new features that would add this information in the data.

# %%
X = np.vstack([x, x ** 2, x ** 3]).T

linear_regression.fit(X, y)

grid_augmented = np.vstack([grid, grid ** 2, grid ** 3]).T
y_pred = linear_regression.predict(grid_augmented)

plt.plot(grid, y_pred, linewidth=3)
plt.scatter(x, y, color="k", s=9)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, linear_regression.predict(X))
print(f"Lowest mean squared error = {mse:.2f}")

# %% [markdown]
# We can see that even with a linear model, we overcome the limitation of the
# model by adding the non-linearity component into the design of additional
# features. Here, we created new feature by knowing the way the target was
# generated. In practice, this is usually not the case. Instead, one is usually
# creating interaction between features with different order, at risk of
# creating a model with too much expressivity and wich might overfit. In
# scikit-learn, the `PolynomialFeatures` is a transformer to create such
# feature interactions which we could have used instead of creating new 
# features ourself.
#

# To present the `PolynomialFeatures`, we are going to use a scikit-learn 
# pipeline which will first create the new features and then fit the model.
# We will later comeback to details regarding scikit-learn pipeline.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

X = x.reshape(-1, 1)

model = make_pipeline(
    PolynomialFeatures(degree=3), LinearRegression()
)
model.fit(X, y)
y_pred = model.predict(grid.reshape(-1, 1))

plt.plot(grid, y_pred, linewidth=3)
plt.scatter(x, y, color="k", s=9)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, model.predict(X))
print(f"Lowest mean squared error = {mse:.2f}")

# %% [markdown]
# Thus, we saw that the `PolynomialFeatures` is actually doing the same
# operation that we did manually above.

# %% [markdown]
# **FIXME: it might be to complex to be introduced here but it seems good in
# the flow. However, we go away from linear model.**
#
# The last possibility to make a linear model more expressive is to use
# "kernel". Instead of learning a weight per feature as we previously
# emphasized, a weight will be assign by sample instead. However, not all
# sample will be used. This is the base of the support vector machine
# algorithm.

# %%
from sklearn.svm import SVR

svr = SVR(kernel="linear").fit(X, y)
y_pred = svr.predict(grid.reshape(-1, 1))

plt.plot(grid, y_pred, linewidth=3)
plt.scatter(x, y, color="k", s=9)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, svr.predict(X))
print(f"Lowest mean squared error = {mse:.2f}")

# %% [markdown]
# The algorithm can be modified such that it can use non-linear kernel. Then,
# it will compute interaction between samples using this non-linear
# interaction.

svr = SVR(kernel="poly", degree=3).fit(X, y)
y_pred = svr.predict(grid.reshape(-1, 1))

plt.plot(grid, y_pred, linewidth=3)
plt.scatter(x, y, color="k", s=9)
plt.xlabel("x", size=26)
plt.ylabel("y", size=26)

mse = mean_squared_error(y, svr.predict(X))
print(f"Lowest mean squared error = {mse:.2f}")

# %% [markdown]
# Therefore, kernel can make a model more expressive.

# %% [markdown]
# ### Linear regression in higher dimension
# In the previous example, we usually used only a single feature. But we
# already shown that we could add new feature to make the model more expressive
# by deriving this new features based on the original feature.
#
# Indeed, we could also use additional features which are decorrelated with the
# original feature and that could help us to predict the target.
#
# We will load a dataset reporting the median house value in California.
# The dataset in made of 8 features regarding demography and geography of the
# location and the aim is to predict the median house price.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X.head()

# %% [markdown]
# We will compare the score of `LinearRegression` and `Ridge` (which is a
# regularized version of linear regression).
#
# We will evaluate our model using the mean squared error as in the previous
# example. The lower the score, the better.

# %% [markdown]
# Here we will divide our data into a training set and a validation set.
# The validation set will be used to assert the hyper-parameters selection.
# While a testing set should only be used to assert the score of our final
# model.

# %%
from sklearn.model_selection import train_test_split

X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, random_state=1
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, random_state=1
)

# %% [markdown]
# Note that in the first example, we did not care about scaling our data to
# keep the original units and have better intuition. However, this is a good
# practice to scale the data such that each feature have a similar standard
# deviation. It will be even more important if the solver used by the model
# is a gradient-descent-based solver.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# %% [markdown]
# Scikit-learn provides several tools to preprocess the data. The
# `StandardScaler` transforms the data such that each feature will have a zero
# mean and a unit standard deviation.
#
# These scikit-learn estimators are known are transformer: they compute some
# statistics and store them when calling `fit`. Using these statistics, they
# transform the data when calling `transform`. Therefore, it is important to
# note that `fit` should only be called on the training data similarly to the
# classifier or regressor.
#
# In the example above, `X_train_scaled` are data scaled after computing the
# mean and standard deviation of each feature considering the training data.
# `X_test_scaled` are data scaled using the mean and standard deviation of each
# feature on the training data.

# %%
linear_regression = LinearRegression()
linear_regression.fit(X_train_scaled, y_train)
y_pred = linear_regression.predict(X_valid_scaled)
print(
    f"Mean squared error on the validation set: "
    f"{mean_squared_error(y_valid, y_pred):.4f}"
)

# %% [markdown]
# Instead of manually transforming the data by calling the transformer,
# scikit-learn provide a `Pipeline` allowing to call a sequence of
# transformer(s) followed by a regressor or a classifier. This pipeline exposed
# the same API than the regressor and classifier and will manage the call to
# `fit` and `transform` for you, avoiding any mistake with data leakage.
#
# We already presented `Pipeline` in the second notebook and we will use it
# here to combine both the scaling and the linear regression.
#
# We will call `make_pipeline()` which will create a `Pipeline` by giving as
# arguments the successive transformations to perform followed by the regressor
# model.
#
# So the two cells above become this new one:

# %%
from sklearn.pipeline import make_pipeline

linear_regression = make_pipeline(StandardScaler(), LinearRegression())

linear_regression.fit(X_train, y_train)
y_pred_valid = linear_regression.predict(X_valid)
linear_regression_score = mean_squared_error(y_valid, y_pred_valid)
y_pred_test = linear_regression.predict(X_test)

print(
    f"Mean squared error on the validation set: "
    f"{mean_squared_error(y_valid, y_pred_valid):.4f}"
)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)

# %% [markdown]
# Now we want to compare this basic `LinearRegression` versus its regularized
# form `Ridge`.
#
# We will tune the parameter alpha and compare it with the `LinearRegression`
# model which is not regularized.

# %%
from sklearn.linear_model import Ridge

ridge = make_pipeline(StandardScaler(), Ridge())

list_alphas = np.logspace(-2, 2.1, num=40)
list_ridge_scores = []
for alpha in list_alphas:
    ridge.set_params(ridge__alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_valid)
    list_ridge_scores.append(mean_squared_error(y_valid, y_pred))

plt.plot(
    list_alphas, [linear_regression_score] * len(list_alphas), '--',
    label='LinearRegression',
)
plt.plot(list_alphas, list_ridge_scores, "+-", label='Ridge')
plt.xlabel('alpha (regularization strength)')
plt.ylabel('Mean squared error (lower is better')
_ = plt.legend()

# %% [markdown]
# We see that, just like adding salt in cooking, adding regularization in our
# model could improve its error on the validation set. But too much
# regularization, like too much salt, decrease its performance.
#
# We can see visually that the best alpha should be around 40.

# %%
best_alpha = list_alphas[np.argmin(list_ridge_scores)]
best_alpha

# %% [markdown]
# At the end, we selected this alpha *without* using the testing set ; but 
# instead by extracting a validation set which is a subset of the training
# data. This has been seen in the lesson *basic hyper parameters tuning*.
# We can finally compared the `LinearRegression` model and the best `Ridge`
# model on the testing set.

# %%
print("Linear Regression")
y_pred_test = linear_regression.predict(X_test)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)

print("Ridge Regression")
ridge.set_params(ridge__alpha=alpha)
ridge.fit(X_train, y_train)
y_pred_test = ridge.predict(X_test)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)
# FIXME add explication why Ridge is not better (equivalent) than linear 
# regression here.

# %% [markdown]
# The hyperparameter search could have been made using the `GridSearchCV`
# instead of manually splitting the training data and selecting the best alpha.

# %%
from sklearn.model_selection import GridSearchCV

ridge = GridSearchCV(
    make_pipeline(StandardScaler(), Ridge()),
    param_grid={"ridge__alpha": list_alphas},
)
ridge.fit(X_train_valid, y_train_valid)
print(ridge.best_params_)

# %% [markdown]
# The `GridSearchCV` manage to test all possible given `alpha` value and picked
# up the best one with a cross-validation scheme. We can now compare with
# the `LinearRegression`.

# %%
print("Linear Regression")
linear_regression.fit(X_train_valid, y_train_valid)
y_pred_test = linear_regression.predict(X_test)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)

print("Ridge Regression")
y_pred_test = ridge.predict(X_test)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)

# %% [markdown]
# It is as well interesting to know that several regressors and classifiers
# in scikit-learn are optimized to make this parameter tuning. They usually
# finish with the term "CV" for "Cross Validation" (e.g. `RidgeCV`).
# They are more efficient than making the `GridSearchCV` by hand and you
# should use them instead.
#
# We will repeat the equivalent of the hyper-parameter search but instead of
# using a `GridSearchCV`, we will use `RidgeCV`.

# %%
from sklearn.linear_model import RidgeCV

ridge = make_pipeline(
    StandardScaler(), RidgeCV(alphas=[.1, .5, 1, 5, 10, 50, 100])
)
ridge.fit(X_train_valid, y_train_valid)
ridge[-1].alpha_

# %%
print("Linear Regression")
y_pred_test = linear_regression.predict(X_test)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)

print("Ridge Regression")
y_pred_test = ridge.predict(X_test)
print(
    f"Mean squared error on the test set: "
    f"{mean_squared_error(y_test, y_pred_test):.4f}"
)

# %% [markdown]
# Note that the best parameter value is changing because the cross-validation
# between the different approach is internally different.

# %% [markdown]
# ## 2. Classification
# In regression, we saw that the target to be predicted was a continuous
# variable. In classification, this target will be discrete. (e.g. categorical)
#
# We will go back to our penguin dataset. However, this time we will try to
# predict the penguin species using the culmen information. We will also
# simplify our classification problem by selecting only 2 of the penguin
# species to solve a binary classification problem.

# %%
data = pd.read_csv("../datasets/penguins.csv")

# select the features of interest
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

data = data[culmen_columns + [target_column]]
data[target_column] = data[target_column].str.split().str[0]
data = data[data[target_column].apply(lambda x: x in ("Adelie", "Chinstrap"))]
data = data.dropna()

# %% [markdown]
# We can quickly start by visualizing the feature distribution by class

# %%
_ = sns.pairplot(data=data, hue="Species")

# %% [markdown]
# So we can observe, that we have quite a simple problem. When the culmen
# length increase, the probability to be a Chinstrap penguin is closer to 1.
# However, the culmen length does not help at predicting the penguin specie.
#
# For the later model fitting, we will separate the target from the data and
# we will create a training and a testing set.

# %%
from sklearn.model_selection import train_test_split

X, y = data[culmen_columns], data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)

# %% [markdown]
# To visualize the separation found by our classifier, we will define an helper
# function `plot_decision_function` . In short, this function will fit our classifier and
# plot the edge of the decision function, where the probability to be an Adelie or
# Chinstrap will be equal (p=0.5).


# %%
def plot_decision_function(X, y, clf, title="auto", ax=None):
    """Plot the boundary of the decision function of a classifier."""
    from sklearn.preprocessing import LabelEncoder

    clf.fit(X, y)

    # create a grid to evaluate all possible samples
    plot_step = 0.02
    feature_0_min, feature_0_max = (
        X.iloc[:, 0].min() - 1,
        X.iloc[:, 0].max() + 1,
    )
    feature_1_min, feature_1_max = (
        X.iloc[:, 1].min() - 1,
        X.iloc[:, 1].max() + 1,
    )
    xx, yy = np.meshgrid(
        np.arange(feature_0_min, feature_0_max, plot_step),
        np.arange(feature_1_min, feature_1_max, plot_step),
    )

    # compute the associated prediction
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    # make the plot of the boundary and the data samples
    if ax is None:
        _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(
        data=pd.concat([X, y], axis=1),
        x=X.columns[0],
        y=X.columns[1],
        hue=y.name,
        ax=ax,
    )
    if title == "auto":
        C = clf[-1].C if hasattr(clf[-1], "C") else clf[-1].C_
        ax.set_title(f"C={C}\n with coef={clf[-1].coef_[0]}")
    else:
        ax.set_title(title)


# %% [markdown]
# ### Un-penalized logistic regression
#
# The linear regression that we previously saw will predict a continuous
# output. When the target is a binary outcome, one can use the logistic
# function to model the probability. This model is known as logistic
# regression.
#
# Scikit-learn provides the class `LogisticRegression` which implement this
# algorithm.

# %%
from sklearn.linear_model import LogisticRegression

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="none")
)
plot_decision_function(X_train, y_train, logistic_regression)

# %% [markdown]
# Thus, we see that our decision function is represented by a line separating
# the 2 classes. Since the line is oblique, it means that we used a
# combination of both features:

# %%
print(logistic_regression[-1].coef_)

# %% [markdown]
# Indeed, both coefficients are non-null.
#
# ### Apply some regularization when fitting the logistic model
#
# The `LogisticRegression` model
# allows to apply regularization via the parameter `C`. It would be equivalent
# to shift from `LinearRegression` to `Ridge`. On the contrary to `Ridge`, the
# `C` parameter is the inverse of the regularization strength: a smaller `C`
# will lead to a more regularized model. We can check the effect of
# regularization on our model:

# %%
_, axs = plt.subplots(ncols=3, figsize=(12, 4))

for ax, C in zip(axs, [0.02, 0.1, 1]):
    logistic_regression = make_pipeline(
        StandardScaler(), LogisticRegression(C=C)
    )
    plot_decision_function(
        X_train, y_train, logistic_regression, ax=ax,
    )

# %% [markdown]
# A more regularized model will make the coefficients tend to 0. Since one of
# the feature is considered less important when fitting the model (lower
# coefficient magnitude), only one of the feature will be used when C is small.
# This feature is the culmen length which is in line with our first insight
# that we found when plotting the marginal feature probabilities.
#
# Just like the `ridgeCV` class which automatically find the optimal `alpha`, 
# one can use `LogisticRegressionCV` to find the best `C` on the training data.

# %%
from sklearn.linear_model import LogisticRegressionCV

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10])
)
plot_decision_function(X_train, y_train, logistic_regression)

# %% [markdown]
# ### Beyond linear separation
#
# As we saw in regression, the linear classification model expects the data
# to be linearly separable. When this assumption does not hold, the model
# is not enough expressive to properly fit the data. One need to apply the same
# tricks than in regression: feature augmentation (using expert-knowledge
# potentially) or using method based on kernel.
#
# We will provide examples where we will use a kernel support vector machine
# to make classification on some toy-dataset where it is impossible to find a perfect linear
# separation

# %%
from sklearn.datasets import (
    make_moons, make_classification, make_gaussian_quantiles,
)

X_moons, y_moons = make_moons(n_samples=500, noise=.13, random_state=42)
X_class, y_class = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    random_state=2,
)
X_gauss, y_gauss = make_gaussian_quantiles(
    n_samples=50, n_features=2, n_classes=2, random_state=42,
)

datasets = [
    [pd.DataFrame(X_moons, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_moons, name="class")],
    [pd.DataFrame(X_class, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_class, name="class")],
    [pd.DataFrame(X_gauss, columns=["Feature #0", "Feature #1"]),
     pd.Series(y_gauss, name="class")],
]

# %%
from sklearn.svm import SVC

_, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 9))

linear_model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
kernel_model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

for ax, (X, y) in zip(axs[0], datasets):
    plot_decision_function(X, y, linear_model, title="Linear kernel", ax=ax)
for ax, (X, y) in zip(axs[1], datasets):
    plot_decision_function(X, y, kernel_model, title="RBF kernel", ax=ax)

# %% [markdown]
# We see that the $R^2$ score decrease on each dataset, so we can say that each
# dataset is "less linearly separable" than the previous one.

# %% [markdown]
# # Main take away
#
# - `LinearRegression` find the best slope which minimize the mean squared
#   error on the train set
# - `Ridge` could be better on the test set, thanks to its regularization
# - `RidgeCV` and `LogisiticRegressionCV` find the best relugarization thanks
#   to cross validation on the training data
# - `pipeline` can be used to combinate a scaler and a model
# - If the data are not linearly separable, we shall use a more complex model
#   or use feature augmentation
#

# %%
