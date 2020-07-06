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
# In this notebook we will review linear model from `sklearn`.
# We will : 
# - fit a simple linear slope 
# - use `LinearRegression` and its regularized version `Ridge` which is more robust.
# - use `LogisticRegression` on the dataset "adult census".
# - see examples of linear separability

# %% [markdown]
# ## 1. Linear regression

# %% [markdown]
# Before loading any dataset, we will first explore a very simple linear model: our data is one dimensional, and the target value is continuous. 
# For instance our `x` might be the years of experiences (normalized) and `y` the salary (normalized).

# %%
import numpy as np
import matplotlib.pyplot as plt

## data generation
# fix the seed for reproduction
np.random.seed(0)

n_sample = 100
x_max, x_min = 1.4, -1.4
len_x = (x_max - x_min)
x = np.random.rand(n_sample) * len_x - len_x/2
noise = np.random.randn(n_sample) * .3
y = x**3 -.5*x**2 +  noise

## plot the data
plt.scatter(x,y,  color='k', s=9)
plt.xlabel('x', size=26)
plt.ylabel('y', size=26)


# %% [markdown]
# ### Exercice 1
#
# In this exercice, you are asked to approximate the target `y` by a linear function `f(x)`. i.e. find the best coefficients of the function `f` in order to minimize the error.
#
# Then you could compare the mean squared error of your model with the mean squared error of a linear model, which shall be the minimal one.

# %%
def f(x):
    w0 = 0 # TODO: update the weight here
    w1 = 0 # TODO: update the weight here
    y_predict = w1 * x + w0
    return y_predict

# plot the slope of f
grid = np.linspace(x_min, x_max, 300)
plt.scatter(x, y, color='k', s=9)
plt.plot(grid, f(grid), linewidth=3)

error_mse = np.sqrt(np.mean((y - f(x))**2))
print(f'Mean squared error = {error_mse}')

# %%
# Solution 1. by fiting a linear regression

from sklearn import linear_model

lr = linear_model.LinearRegression()
# X should be 2D for sklearn
X = x.reshape((-1,1))
lr.fit(X,y)

# plot the best slope
y_best = lr.predict(grid.reshape(-1, 1))
plt.plot(grid, y_best, linewidth=3)
plt.scatter(x, y, color='k', s=9)

error_mse = np.sqrt(np.mean(((y - lr.predict(X))**2)))
print(f'Lowest mean squared error = {error_mse}')
print(f'best coef: w1 = {lr.coef_[0]}, best intercept: w0 = {lr.intercept_}')

# %% [markdown]
# ### Linear regression in higher dimension
#
# We will now load a new dataset from the “Current Population Survey” from 1985 to predict the **Salary** as a function of various features such as *experience, age*, or *education*.
# For simplicity, we will only use this numerical features.
#
# We will compare the score of the `linear regression` and the `ridge regression` (which is simply a regularized version of the linear regression).
# Here the score will be the $R^2$ score, which is the score by default of a Rergessor. It represents the proportion of variance of the target explained by the model. The best score possible is 1.
#

# %%
from sklearn.datasets import fetch_openml

# Load the data
survey = fetch_openml(data_id=534, as_frame=True)
X = survey.data[survey.feature_names]
y = np.log(survey.target)
numerical_columns = ['EDUCATION', 'EXPERIENCE', 'AGE']
X = X[numerical_columns]
X.head()

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

# fit linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)

# %%
list_ridge_scores = []
list_alphas = np.logspace(-2,2.2)#[.01, .1, 1, 10, 100]

for alpha in list_alphas:
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    list_ridge_scores.append(ridge.score(X_test, y_test))
    
plt.plot(list_alphas, [lr_score] * len(list_alphas), '--', label = 'LinearRegression', linewidth = 3)
plt.plot(list_alphas, list_ridge_scores, label = 'Ridge', linewidth = 3)
plt.xlabel('alpha (regularization strength)', size=16)
plt.ylabel('$R^2$ Score (higher is better)', size = 16)
_ = plt.legend()

# %% [markdown]
# We see that, just like adding salt in cooking, adding regularization in our model could improve its error on the test set. But too much regularization, like too much salt, decrease its performance.
#
# Fortunatly, the `sklearn` api provides us with an automatic way to find the best regularization `alpha` with the module `RidgeCV`. For that, it internaly computes a cross validation on the training set to predict the best `alpha` parameter.

# %%
from sklearn.linear_model import RidgeCV

ridge = RidgeCV()
ridge.fit(X_train, y_train)

lr_score = lr.score(X_test, y_test)
print(f'R2 score of linear regression  = {lr_score}')
print(f'R2 score of ridgeCV regression = {ridge.score(X_test, y_test)}')
print(f'best alpha found = {ridge.alpha_}')

# %% [markdown]
# ## 2. Logistic regresion 

# %% [markdown]
# We will load the "adult census" dataset, already used in previous notebook.
# The class to predict is either a person earn more than $50k per year.

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
# "i" denotes integer type, "f" denotes float type
numerical_columns = [
    col for col in data.columns
    if df[col].dtype.kind in ["i", "f"]]
numerical_columns

data_numeric = df[numerical_columns]
data_numeric.head()

# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    data_numeric, target, random_state=42)

# %% [markdown]
# As seen in previous notebook, it is always a good idea to scale the input data when we are using a linear model.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# `LogisticRegression` comes already with a build-in regulartization parameters `C`. 
#
# Contrary to `alpha` in Ridge, the parameters `C` here is the inverse of regularization strength;  so smaller values specify stronger regularization.
#
# Here we will fit `LogisiticRegressionCV` to get the best regularization parameters`C` on the training set.

# %%
from sklearn.linear_model import LogisticRegressionCV

log_reg = LogisticRegressionCV()
log_reg.fit(X_train_scaled,y_train)

# %% [markdown]
# The default score in `LogisticRegression` is the accuracy.

# %%
log_reg.score(X_test_scaled, y_test)

# %% [markdown]
# ## 3. Linear separability

# %%
from sklearn.datasets import make_blobs, make_moons, make_classification, make_gaussian_quantiles
from sklearn.linear_model import LogisticRegression

def plot_linear_separation(X, y):
    # plot the separation line from a logistic regression
    
    clf = LogisticRegression()
    clf.fit(X,y)
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = - (clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,0])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, linewidths = 3, levels = 0) 
    plt.title(f'$R^2$ score: {clf.score(X,y)}')


# %%
X_blobs, y_blobs = make_blobs(n_samples = 500, n_features=2, centers=[[3,3],[0,8]], random_state = 42)
X_moons, y_moons = make_moons(n_samples= 500, noise=.13, random_state = 42)
X_class, y_class = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=2)
X_gauss, y_gauss = make_gaussian_quantiles(n_features=2, n_classes=2, random_state = 42)

list_data = [[X_blobs, y_blobs],
            [X_moons, y_moons],
            [X_class, y_class],
            [X_gauss, y_gauss]]

for X,y in list_data:
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y,
                s=50, edgecolor='k')
    plot_linear_separation(X, y)
    

# %% [markdown]
# # Main take away
#
# - Linear regression find the best slope to minimize the mean squared error on the train set
# - Ridge regression could be better on the test set, thanks to its regularization
# - RidgeCV and LogisiticRegressionCV find the best relugarization thanks to cross validation
# - If the data are not linearly separable, we shall use a more complex model
#

# %%
