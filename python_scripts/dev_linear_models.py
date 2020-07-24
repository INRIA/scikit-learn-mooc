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
# In this notebook we will review some linear models from `scikit-learn`.
# We will :
# - fit a simple linear slope;
# - use `LinearRegression` and its regularized version `Ridge` which is more
#   robust;
# - use `LogisticRegression` on the dataset "adult census" with `pipeline`;
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
plt.ylabel('y', size=26)

# %% [markdown]
# ### Exercice 1
#
# In this exercice, you are asked to approximate the target `y` by a linear
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

mse = mean_squared_error(y, f(x))
print(f'Mean squared error = {mse}')

# %% [markdown]
# ### Solution 1. by fiting a linear regression

# %%
from sklearn import linear_model

lr = linear_model.LinearRegression()
# X should be 2D for sklearn
X = x.reshape((-1, 1))
lr.fit(X, y)

# plot the best slope
y_best = lr.predict(grid.reshape(-1, 1))
plt.plot(grid, y_best, linewidth=3)
plt.scatter(x, y, color='k', s=9)

mse = mean_squared_error(y, lr.predict(X))
print(f'Lowest mean squared error = {mse}')

# %% [markdown]
# Here the coeficients learnt by `LinearRegression` is the best slope which fit
# the data. We can inspect those coeficents using the attributes of the model
# learnt as follow:

# %%
print(f'best coef: w1 = {lr.coef_[0]}, best intercept: w0 = {lr.intercept_}')

# %% [markdown]
# ### Linear regression in higher dimension
#
# We will now load a new dataset from the “Current Population Survey” from 1985
# to predict the **salary** as a function of various features such as
# *experience, age*, or *education*.
# For simplicity, we will only use this numerical features.
#
# We will compare the score of `LinearRegression` and `Ridge` (which is a
# regularized version of linear regression).
# Here the score will be the $R^2$ score, which is the score by default of a
# Rergessor. It represents the proportion of variance of the target explained
# by the model. The best $R^2$ score possible is 1.

# %%
from sklearn.datasets import fetch_openml

# Load the data
survey = fetch_openml(data_id=534, as_frame=True)
X = survey.data[survey.feature_names]
y = np.log(survey.target)
numerical_columns = ['EDUCATION', 'EXPERIENCE', 'AGE']
X = X[numerical_columns]
X.head()

# %% [markdown]
# Here we will divide our data into a training set and a validation set.
# The validation set will be used to assert the hyper-parameters selection.
# While a testing set should only be used to assert the score of our final
# model.

# %%
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=5, random_state=1
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, random_state=1
)

# %% [markdown]
# Since the data are not scaled, we should scale them before applying our
# linear model.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# fit linear regression
linear_regression = LinearRegression()
linear_regression.fit(X_train_scaled, y_train)
linear_regression_score = linear_regression.score(X_test_scaled, y_test)

# %% [markdown]
# As seen during the second notebook, we will use the scikit-learn `Pipeline`
# module to combine both the scaling and the linear regression.
#
# Using pipeline is more convenient and it is safer: it avoids leaking
# statistics from the validation set or the testing set into the trained model.
#
# We will call `make_pipeline()` which will create a `Pipeline` by giving as
# arguments the successive transformations to perform followed by the regressor
# model.
#
# So the two cells above become this new one:

# %%
from sklearn.pipeline import make_pipeline

model_linear = make_pipeline(StandardScaler(),
                             LinearRegression())
model_linear.fit(X_train, y_train)
linear_regression_score = model_linear.score(X_valid, y_valid)

# %% [markdown]
# Now we want to compare this basic `LinearRegression` versus its regularized
# form `Ridge`.
#
# We will present the score on the validation set for different values of
# `alpha`, which controls the regularization strength in `Ridge`.

# %%
# taking the alpha between .01 and 100,
# spaced evenly on a log scale.
list_alphas = np.logspace(-2, 2)

list_ridge_scores = []
for alpha in list_alphas:
    # fit Ridge
    ridge = make_pipeline(
        StandardScaler(), Ridge(alpha=alpha)
    )
    ridge.fit(X_train, y_train)
    list_ridge_scores.append(ridge.score(X_valid, y_valid))

plt.plot(
    list_alphas, [linear_regression_score] * len(list_alphas), '--',
    label='LinearRegression', linewidth=3
)
plt.plot(list_alphas, list_ridge_scores, label='Ridge', linewidth=3)
plt.xlabel('alpha (regularization strength)', size=16)
plt.ylabel('$R^2$ Score (higher is better)', size=16)
_ = plt.legend()

# %% [markdown]
# We see that, just like adding salt in cooking, adding regularization in our
# model could improve its error on the validation set. But too much
# regularization, like too much salt, decrease its performance.
# In our case, the alpha parameters is best when is around 20.
#
# Note that the calibration of `alpha` could not be tuned on the test set -
# otherwise we are fitting the test set, which would correspond to overfitting.
#
# To calibrate `alpha` on our training set, we have extract a small
# validation set from our training set. That is seen on the lesson:
# *basic hyper parameters tuning*.
#
# Fortunatly, the `scikit-learn` api provides us with an automatic way to find
# the best regularization `alpha` with the module `RidgeCV`. For that, it
# internaly computes a cross validation on the training set (with a validation
# set) to predict the best `alpha` parameter.

# %%
from sklearn.linear_model import RidgeCV

ridge = make_pipeline(
    StandardScaler(), RidgeCV(alphas=[.1, .5, 1, 5, 10, 50, 100])
)
# tune alpha on the traingin set
ridge.fit(X_train, y_train)

linear_regression_score = linear_regression.score(X_test, y_test)
print(f'R2 score of linear regression  = {linear_regression_score}')
print(f'R2 score of ridgeCV regression = {ridge.score(X_test, y_test)}')
print(f'The best `alpha` found on the training set is {ridge[1].alpha_}')

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
# ## Feature augmentation

# %% [markdown]
# Let consider a toy dataset, where the target is a function of both `x` and
# `sin(x)`.
# In this case, a linear model will only fit the linear part.

# %%
n_samples = 100
x = np.arange(0, 10, 10 / n_samples)
noise = np.random.randn(n_samples)
y = 1.5 * np.sin(x) + x + noise
X = x.reshape((-1, 1))

linear_regression = LinearRegression()
linear_regression.fit(X, y)
y_predict_linear = linear_regression.predict(X)
plt.scatter(X, y)
plt.plot(
    X, y_predict_linear, label='predict with linear', color='k', linewidth=3
)

# %% [markdown]
# Now, if we want to extend the power of expression of our model, we could add
# whatever combination of the feature, to enrich the feature space, thus
# enriching the complexity of the model.

# %%
X_augmented = np.concatenate((X, np.sin(X)), axis=1)
linear_regression = LinearRegression()
linear_regression.fit(X_augmented, y)
y_predict_augmented = linear_regression.predict(X_augmented)
plt.scatter(X, y)
plt.plot(
    X, y_predict_linear, label='predict with linear', color='k', linewidth=3
)
plt.plot(
    X, y_predict_augmented, label='predict with augmented', color='orange',
    linewidth=4
)

plt.legend()

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
