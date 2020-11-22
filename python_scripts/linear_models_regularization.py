# %% [markdown]
# # Regularization of linear regression model
#
# In this notebook, we will see the limitation of linear regression model and
# the advantage of using regularized models instead. Besides, we will also
# present the preprocessing required when dealing with regularized model,
# furthermore when the regularization parameter needs to be fine tuned.
#
# We will start by highlighting the over-fitting issue that can arise with
# a simple linear regression model.
#
# ## Effect of regularization
#
# We will first load the california housing dataset.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X.head()

# %% [markdown]
# As in the previous exercise, we will use an independent test set to evaluate
# the performance of our model.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.5
)

# %% [markdown]
# In one of the previous notebook, we show that linear model could be used
# even in setting where `X` and `y` are not linearly linked. We showed that one
# can use the `PolynomialFeatures` transformer to create new feature encoding
# non-linear interactions between features. Here, we will use this transformer
# to augment the feature space. Subsequently, we train a linear regresion
# model. We will use the out-of-sample test set to evaluate the generalization
# capabilities of our model.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

linear_regression = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
linear_regression.fit(X_train, y_train)

print(
    f"R2 score of linear regresion model on the test set:\n"
    f"{linear_regression.score(X_test, y_test):.3f}"
)

# %% [markdown]
# We see that we obtain an $R^2$ score below zero. It means that our model is
# far worth than predicting the mean of `y_train`. This is issue is due to
# overfitting. We can compute the score on the training set to confirm this
# intuition.

# %%
print(
    f"R2 score of linear regresion model on the train set:\n"
    f"{linear_regression.score(X_train, y_train):.3f}"
)

# %% [markdown]
# The score on the training set is much better. This performance gap between
# the training and testing score is an indication that our model overfitted
# our training set.
#
# Indeed, this is one of the danger when augmenting the number of features
# with a `PolynomialFeatures` transformer. Our model will focus on some
# specific features. We can check the weights of the model to have a
# confirmation.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

weights_linear_regression = pd.Series(
    linear_regression[-1].coef_,
    index=linear_regression[0].get_feature_names(
        input_features=X.columns)
)
_, ax = plt.subplots(figsize=(6, 16))
_ = weights_linear_regression.plot(kind="barh", ax=ax)

# %% [markdown]
# We can force the linear regression model to consider all features in a more
# homogeneous manner. In fact, we could force large positive or negative weight
# to shrink toward zero. This is known as regularization. We will use a ridge
# model which enforce such behaviour.

# %%
from sklearn.linear_model import Ridge

ridge = make_pipeline(
    PolynomialFeatures(degree=2),
    Ridge(alpha=0.5)
)
ridge.fit(X_train, y_train)

# %%
print(
    f"R2 score of ridge model on the train set:\n"
    f"{ridge.score(X_train, y_train):.3f}"
)

# %%
print(
    f"R2 score of ridge model on the test set:\n"
    f"{ridge.score(X_test, y_test):.3f}"
)

# %% [markdown]
# We see that the training and testing score are much closer, indicating that
# our model is less overfitting. We can compare the values of the weights of
# ridge with the un-regularized linear regression.

# %%
weights_ridge = pd.Series(
    ridge[-1].coef_,
    index=ridge[0].get_feature_names(
        input_features=X.columns)
)

# %%
weights = pd.concat(
    [weights_linear_regression, weights_ridge], axis=1,
    keys=["Linear regression", "Ridge"])

_, ax = plt.subplots(figsize=(6, 16))
weights.plot(kind="barh", ax=ax)

# %% [markdown]
# We see that the magnitude of the weights are shrinked towards zero in
# comparison with the linear regression model.
#
# However, in this example, we omitted two important aspects: (i) the need to
# scale the data and (ii) the need to search for the best regularization
# parameter.
#
# ## You shall scale your data
#
# Regularization will add constraints on weights of the model. We saw in the
# previous example that a ridge model will enforce that all weights to have a
# similar magnitude. Indeed, larger is alpha, larger will be this enforcement.
#
# This procedure should make us think about feature rescaling. Let's think
# about the case where features have an identical data dispersion, if two
# features are found equally important by the model, they will be affected
# weights close in term of norm.
#
# Now, let's think about the scenario but where features will have completely
# different data dispersion (e.g. age of person in year and it annual revenue
# in $). If two features are as important, our model will boost the weight of
# feature with small dispersion and reduce the weight of the feature with high
# dispersion. We recall that regularization force weights to be closer.
#
# Therefore, we get an intuition that if we want to use regularization, dealing
# with rescaled data would make it easier to find an optimal regularization
# parameter and thus an adequate model. As a side note, some solver based on
# gradient computation are expecting such rescaled data. Unscaled data will be
# detrimental when computing the optimal weights. Therefore, when working with
# a linear model and numerical data, this is in general a good practice to
# scale the data.
#
# In the remaining of this section, we will present the basics on how to
# incorporate a scaler within your machine learning pipeline. Scikit-learn
# provides several tools to preprocess the data. The `StandardScaler`
# transforms the data such that each feature will have a mean of zero and a
# standard deviation of 1.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# This scikit-learn estimator is known as a transformer: it computes some
# statistics (i.e the mean and the standard deviation) and stores them as
#  attributes (scaler.mean_, scaler.scale_)
# when calling `fit`. Using these statistics, it
# transform the data when `transform` is called. Therefore, it is important to
# note that `fit` should only be called on the training data, similar to
# classifiers and regressors.

# %%
print('mean records on the training set:\n', scaler.mean_)
print('standard deviation records on the training set:\n', scaler.scale_)

# %% [markdown]
# In the example above, `X_train_scaled` is the data scaled, using the
# mean and standard deviation of each feature, computed using the training
# data `X_train`.
#
# Thus, we can use these scaled dataset to train and test our model.

# %%
ridge.fit(X_train_scaled, y_train)

print(
    f"R2 score of ridge model on the test set:\n"
    f"{ridge.score(X_test_scaled, y_test):.3f}"
)

# %% [markdown]
# Instead of calling the transformer to transform the data and then calling the
# regressor, scikit-learn provides a `Pipeline`, which 'chains' the transformer
# and regressor together. The pipeline allows you to use a sequence of
# transformer(s) followed by a regressor or a classifier, in one call. (i.e.
# fitting the pipeline will fit both the transformer(s) and the regressor. Then
# predicting from the pipeline will first transform the data through the
# transformer(s) then predict with the regressor from the transformed data)
#
# This pipeline exposes the same API as the regressor and classifier and will
# manage the calls to `fit` and `transform` for you, avoiding any problems with
# data leakage (when knowledge of the test data was inadvertently included in
# training a model, as when fitting a transformer on the test data).
#
# We already use such `Pipeline` to create the polynomial features before to
# train the model.
#
# We will can create a `Pipeline` by using `make_pipeline` and giving as
# arguments the transformation(s) to be performed (in order) and the regressor
# model.
#
# Here, we can integrate the scaling phase before to train our model:

# %%
ridge = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    Ridge(alpha=0.5)
)
ridge.fit(X_train, y_train)

print(
    f"R2 score of ridge model on the test set:\n"
    f"{ridge.score(X_test, y_test):.3f}"
)

# %% [markdown]
# In the previous example, we see the benefit of using a pipeline. It
# simplifies the manual handling.
#
# When creating the model, keeping the same `alpha` does not give good results.
# It depends on the data provided. Therefore, it needs to be tuned for each
# dataset.
#
# In the next section, we will present the steps to tune the parameters.
#
# ## Fine tuning the regularization parameter
#
# As mentioned, the regularization parameter needs to be tuned on each dataset.
# The default parameter will not lead to the optimal model. Therefore, we need
# to tune the `alpha` parameter.
#
# Tuning model hyperparameter should be done with care. Indeed, we want to find
# an optimal parameter that maximize some metrics. Thus, it requires a training
# and testing sets. However, this testing set should be different from the
# out-of-sample testing set that we used to evaluate our model. If we use
# the same test, we are using an `alpha` which was optimized for this testing
# set and it breaks the out-of-sample rule.
#
# Therefore, we can split our previous training set into two subsets: a
# new training set and a validation set allowing to later pick the optimal
# alpha.

# %%
X_sub_train, X_valid, y_sub_train, y_valid = train_test_split(
    X_train, y_train, random_state=0, test_size=0.25
)

# %%
import numpy as np

alphas = np.logspace(-10, -1, num=30)
list_ridge_scores = []
for alpha in alphas:
    ridge.set_params(ridge__alpha=alpha)
    ridge.fit(X_sub_train, y_sub_train)
    list_ridge_scores.append(ridge.score(X_valid, y_valid))


plt.plot(alphas, list_ridge_scores, "+-", label='Ridge')
plt.xlabel('alpha (regularization strength)')
plt.ylabel('R2 score (higher is better)')
_ = plt.legend()

# %% [markdown]
# We see that, just like adding salt in cooking, adding regularization in our
# model could improve its error on the validation set. But too much
# regularization, like too much salt, decreases its performance.

# %%
best_alpha = alphas[np.argmax(list_ridge_scores)]
best_alpha

# %% [markdown]
# We can retrain a ridge model on the full training set and set the alpha and
# check the score on the left out dataset.

# %%
ridge = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    Ridge(alpha=best_alpha)
)
ridge.fit(X_train, y_train)

print(
    f"R2 score of ridge model on the test set:\n"
    f"{ridge.score(X_test, y_test):.3f}"
)

# %% [markdown]
# In the next exercise, you will use a scikit-learn estimator which allows to
# make some parameters tuning instead of programming yourself a `for` loop by
# hand.
#
# As a conclusion, you learnt in this notebook about the concept of
# regularization and the importance of preprocessing and parameter tuning.
