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
# - use `LinearRegression` and its regularized version `Ridge` which is more robust;
# - use `LogisticRegression` on the dataset "adult census" with `pipeline`;
# - see examples of linear separability.

# %% [markdown]
# ## 1. Regression: Linear regression

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
# Then you could compare the mean squared error of your model with the mean squared error of a linear model (which shall be the minimal one).

# %%
from sklearn.metrics import mean_squared_error

def f(x):
    w0 = 0 # TODO: update the weight here
    w1 = 0 # TODO: update the weight here
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
#

# %%

from sklearn import linear_model

lr = linear_model.LinearRegression()
# X should be 2D for sklearn
X = x.reshape((-1,1))
lr.fit(X,y)

# plot the best slope
y_best = lr.predict(grid.reshape(-1, 1))
plt.plot(grid, y_best, linewidth=3)
plt.scatter(x, y, color='k', s=9)

mse = mean_squared_error(y, lr.predict(X))
print(f'Lowest mean squared error = {mse}')

# %% [markdown]
# Here the coeficients learnt by `LinearRegression` is the best slope which fit the data.
# We can inspect those coeficents using the attributes of the model learnt as follow:

# %%
print(f'best coef: w1 = {lr.coef_[0]}, best intercept: w0 = {lr.intercept_}')

# %% [markdown]
# ### Linear regression in higher dimension
#
# We will now load a new dataset from the “Current Population Survey” from 1985 to predict the **Salary** as a function of various features such as *experience, age*, or *education*.
# For simplicity, we will only use this numerical features.
#
# We will compare the score of `LinearRegression` and `Ridge` (which is a regularized version of linear regression).
# Here the score will be the $R^2$ score, which is the score by default of a Rergessor. It represents the proportion of variance of the target explained by the model. The best $R^2$ score possible is 1.

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
# As always, we divide our data in a training and in a test set. The test test should only be used to assert the score of our final model.

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

# %% [markdown]
# Since the data are not scaled, we should scale them before applying our linear model.

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
# As seen during the second notebook, we will use the scikit-learn `Pipeline` module to combine both the scaling and the linear regression.
#
# Using pipeline is more convenient and safer (it avoids leaking statistics from the test data into the trained model)  
#
# We will call `make_pipeline()` which will create a `Pipeline` by giving as arguments the successive
# transformations to perform followed by the regressor model.
#
# So the two cells above become this new one :
#

# %%
from sklearn.pipeline import make_pipeline

model_linear = make_pipeline(StandardScaler(),
                             LinearRegression())
model_linear.fit(X_train, y_train)
linear_regression_score = model_linear.score(X_test, y_test)

# %% [markdown]
# Now we want to compare this basic `LinearRegression` versus its regularized form `Ridge`.
#
# We will present the score on the test set for different value of `alpha`, which controls the regularization strength in `Ridge`. 

# %%
# taking the alpha between .001 and 33,
# spaced evenly on a log scale.
list_alphas = np.logspace(-2,1.5)

list_ridge_scores = []
for alpha in list_alphas:
    # fit Ridge
    ridge = make_pipeline(StandardScaler(),
                          Ridge(alpha = alpha))
    ridge.fit(X_train, y_train)
    list_ridge_scores.append(ridge.score(X_test, y_test))
    
plt.plot(list_alphas, [linear_regression_score] * len(list_alphas), '--',
         label = 'LinearRegression', linewidth = 3)
plt.plot(list_alphas, list_ridge_scores, label = 'Ridge', linewidth = 3)
plt.xlabel('alpha (regularization strength)', size=16)
plt.ylabel('$R^2$ Score (higher is better)', size = 16)
_ = plt.legend()

# %% [markdown]
# We see that, just like adding salt in cooking, adding regularization in our model could improve its error on the test set. But too much regularization, like too much salt, decrease its performance.
# In our case, the alpha parameters is best when is around 2.
#
# However, the calibration of `alpha` could not be tuned on the test set - otherwise we are fitting the test set, which would correspond to overfitting.
#
# To calibrate the `alpha` on our training set, we have to extract a small validation set from our training set. That is seen on the lesson : *basic hyper parameters tuning*.
#
# Fortunatly, the `scikit-learn` api provides us with an automatic way to find the best regularization `alpha` with the module `RidgeCV`. For that, it internaly computes a cross validation on the training set to predict the best `alpha` parameter.

# %%
from sklearn.linear_model import RidgeCV

ridge = make_pipeline(StandardScaler(),
                      RidgeCV(alphas = [.1, .5, 1, 5, 10]))
# tune alpha on the traingin set
ridge.fit(X_train_scaled, y_train)

linear_regression_score = linear_regression.score(X_test_scaled, y_test)
print(f'R2 score of linear regression  = {linear_regression_score}')
print(f'R2 score of ridgeCV regression = {ridge.score(X_test_scaled, y_test)}')
print(f'The best `alpha` found on the training set is {ridge[1].alpha_}')

# %% [markdown]
# ## 2. Calssification: Logistic regresion 

# %% [markdown]
# We will load the "adult census" dataset, already used in previous notebook.
# We have to predict either a person earn more than $50k per year or not.

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

# %%
# we will conserve only the numerical features
# "i" denotes integer type, "f" denotes float type
numerical_columns = [
    col for col in data.columns
    if df[col].dtype.kind in ["i", "f"]]
numerical_columns

data_numeric = df[numerical_columns]
data_numeric.head()

# %%
target[:5]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data_numeric, target, random_state=42)

# %% [markdown]
# `LogisticRegression` already comes with a build-in regulartization parameters `C`. 
#
# Contrary to `alpha` in Ridge, the parameters `C` here is the inverse of regularization strength; so smaller values specify stronger regularization.
#
# Here we will fit `LogisiticRegressionCV` to get the best regularization parameters`C` on the training set.
#
# As seen before, we shall scale the input data when using a linear model.

# %%
from sklearn.linear_model import LogisticRegressionCV

model = make_pipeline(StandardScaler(),
                      LogisticRegressionCV())

model.fit(X_train,y_train)

# %% [markdown]
# The default score in `LogisticRegression` is the accuracy.  
# Note that the method `score` of a pipeline is the score of its last estimator.

# %%
model.score(X_test, y_test)

# %% [markdown]
# We then can access each part of the pipeline as accessing a list

# %%
model[1]

# %%
model[1].C_

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
X_class, y_class = make_classification(n_samples= 500, n_features=2, n_redundant=0, n_informative=2, random_state=2)
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
# We see that the $R^2$ score decrease on each dataset, so we can say that each dataset is "less linearly separable" than the previous one.

# %% [markdown]
# ## Feature augmentation

# %% [markdown]
# Let consider a toy dataset, where the target is a function of both `x` and `sin(x)`.
# In this case, a linear model will only fit the linear part.

# %%
n_samples = 100
x = np.arange(0, 10, 10 / n_samples)
noise = np.random.randn(n_samples)
y = 1.5 * np.sin(x) + x + noise
X = x.reshape((-1,1))

linear_regression = LinearRegression()
linear_regression.fit(X, y)
y_predict_linear = linear_regression.predict(X)
plt.scatter(X, y)
plt.plot(X, y_predict_linear, label = 'predict with linear', color = 'k', linewidth = 3)

# %% [markdown]
# Now, if we want to extend the power of expression of our model, we could add whatever combination of the feature, to enrich the feature space, thus enriching the complexity of the model.

# %%
X_augmented = np.concatenate((X, np.sin(X)), axis = 1)
linear_regression = LinearRegression()
linear_regression.fit(X_augmented, y)
y_predict_augmented = linear_regression.predict(X_augmented)
plt.scatter(X, y)
plt.plot(X, y_predict_linear, label = 'predict with linear', color = 'k', linewidth = 3)
plt.plot(X, y_predict_augmented, label = 'predict with augmented', color = 'orange',
         linewidth = 4)

plt.legend()

# %% [markdown]
# # Main take away
#
# - `LinearRegression` find the best slope which minimize the mean squared error on the train set
# - `Ridge` could be better on the test set, thanks to its regularization
# - `RidgeCV` and `LogisiticRegressionCV` find the best relugarization thanks to cross validation on the training data
# - `pipeline` can be used to combinate a scaler and a model
# - If the data are not linearly separable, we shall use a more complex model or use feature augmentation
#
