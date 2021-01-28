# %% [markdown]
# # Regularization of linear regression model
#
# In this notebook, we will see the limitations of linear regression models and
# the advantage of using regularized models instead.
# \
# Besides, we will also present the preprocessing required when dealing
# with regularized models, furthermore when the regularization parameter
# needs to be fine-tuned.
#
# We will start by highlighting the over-fitting issue that can arise with
# a simple linear regression model.
#
# ## Effect of regularization
#
# We will first load the California housing dataset.

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
    X, y, random_state=0, test_size=0.5)

# %% [markdown]
# In one of the previous notebook, we showed that linear models could be used
# even in settings where `X` and `y` are not linearly linked.
# \
# We showed that one can use the `PolynomialFeatures` transformer to create
# additional features encoding non-linear interactions between features.
# \
# Here, we will use this transformer to augment the feature space.
# Subsequently, we will train a linear regression model.
# We will use the out-of-sample test set to evaluate the
# generalization capabilities of our model.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

linear_regression = make_pipeline(PolynomialFeatures(degree=2),
                                  LinearRegression())
linear_regression.fit(X_train, y_train)
test_score = linear_regression.score(X_test, y_test)

print(f"R2 score of linear regresion model on the test set:\n"
      f"{test_score:.3f}")

# %% [markdown]
# We see that we obtain an $R^2$ score below zero.
# \
# It means that our model is far worth than predicting the mean of `y_train`.
# This issue is due to overfitting.
# We can compute the score on the training set to confirm this intuition.

# %%
train_score = linear_regression.score(X_train, y_train)
print(f"R2 score of linear regresion model on the train set:\n"
      f"{train_score:.3f}")

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
    index=linear_regression[0].get_feature_names(input_features=X.columns))
_, ax = plt.subplots(figsize=(6, 16))
_ = weights_linear_regression.plot(kind="barh", ax=ax)

# %% [markdown]
# We can force the linear regression model to consider all features in a more
# homogeneous manner. In fact, we could force large positive or negative weight
# to shrink toward zero. This is known as regularization. We will use a ridge
# model which enforces such behaviour.

# %%
from sklearn.linear_model import Ridge

ridge = make_pipeline(PolynomialFeatures(degree=2),
                      Ridge(alpha=0.5))
ridge.fit(X_train, y_train)

# %%
train_score = ridge.score(X_train, y_train)
print(f"R2 score of ridge model on the train set:\n"
      f"{train_score:.3f}")

# %%
test_score = ridge.score(X_test, y_test)
print(f"R2 score of ridge model on the test set:\n"
      f"{test_score:.3f}")

# %% [markdown]
# We see that the training and testing scores are much closer, indicating that
# our model is less overfitting. We can compare the values of the weights of
# ridge with the un-regularized linear regression.

# %%
weights_ridge = pd.Series(
    ridge[-1].coef_,
    index=ridge[0].get_feature_names(input_features=X.columns))

# %%
weights = pd.concat(
    [weights_linear_regression, weights_ridge], axis=1,
    keys=["Linear regression", "Ridge"])

_, ax = plt.subplots(figsize=(6, 16))
weights.plot(kind="barh", ax=ax)

# %% [markdown]
# We see that the magnitude of the weights are shrunk towards zero in
# comparison with the linear regression model.
#
# However, in this example, we omitted two important aspects: (i) the need to
# scale the data and (ii) the need to search for the best regularization
# parameter.
#
# ## Scale your data!
#
# Regularization will add constraints on weights of the model.
# We saw in the previous example that a ridge model will enforce
# that all weights have a similar magnitude.
# \
# Indeed, the larger alpha is, the larger this enforcement will be.
#
# This procedure should make us think about feature rescaling.
# Let's consider the case where features have an identical data dispersion:
# if two features are found equally important by the model, they will be
# affected close weights in term of norm.
#
# Now, let's consider the scenario where features have completely different
# data dispersion (e.g. age in years and annual revenue in dollars).
# If two features are as important, our model will boost the weights of
# features with small dispersion and reduce the weights of features with
# high dispersion.
# \
# We recall that regularization forces weights to be closer.
#
# Therefore, we get an intuition that if we want to use regularization, dealing
# with rescaled data would make it easier to find an optimal regularization
# parameter and thus an adequate model.
# \
# As a side note, some solvers based on gradient # computation are expecting
# such rescaled data.
# Unscaled data will be detrimental when computing the optimal weights.
# \
# Therefore, when working with a linear model and numerical data,
# it is generally good practice to scale the data.
#
# In the remaining of this section, we will present the basics on how to
# incorporate a scaler within your machine learning pipeline.
# \
# Scikit-learn provides several tools to preprocess the data, such as
# the `StandardScaler`, which transforms the data in order for each feature
# to have a mean of zero and a standard deviation of 1.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# This scikit-learn estimator is known as a transformer: it computes some
# statistics (i.e the mean and the standard deviation) and stores them as
# attributes (`scaler.mean_`, `scaler.scale_`) when calling `fit`. Using these
# stats, it transforms the data when `transform` is called. Therefore, it
# is important to note that `fit` should only be called on the training data,
# similar to classifiers and regressors.

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
test_score = ridge.score(X_test_scaled, y_test)

print(f"R2 score of ridge model on the test set:\n"
      f"{test_score:.3f}")

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
# We already used `Pipeline` to create the polynomial features before training
# the model.
# \
# We will can create a new one by using `make_pipeline` and giving as
# arguments the transformation(s) to be performed (in order) and the regressor
# model.
#
# Here, we can implement the scaling process before training our model:

# %%
ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      Ridge(alpha=0.5))
ridge.fit(X_train, y_train)
test_score = ridge.score(X_test, y_test)

print(f"R2 score of ridge model on the test set:\n"
      f"{test_score:.3f}")

# %% [markdown]
# As we can see in this example, using a pipeline simplifies the manual handling.
#
# When creating the model, keeping the same `alpha` does not give good results.
# It depends on the data provided. Therefore, it needs to be tuned for each
# dataset.
#
# In the next section, we will present the steps to tune this parameter.
#
# ## Fine tuning the regularization parameter
#
# As mentioned, the regularization parameter needs to be tuned on each dataset.
# The default parameter will not lead to the optimal model. Therefore, we need
# to tune the `alpha` parameter.
#
# Model hyperparameters tuning should be done with care. Indeed, we want to find
# an optimal parameter that maximizes some metrics.
# Thus, it requires both a training set and testing set.
# \
# However, this testing set should be different from the out-of-sample testing set
# that we used to evaluate our model:
# if we use the same one, we are using an `alpha` which was optimized for
# this testing set and it breaks the out-of-sample rule.
#
# Therefore, we can split our previous training set into two subsets: a
# new training set and a validation set which we will use later to pick
# the optimal value for `alpha`.

# %%
X_sub_train, X_valid, y_sub_train, y_valid = train_test_split(
    X_train, y_train, random_state=0, test_size=0.25)

# %%
import numpy as np

alphas = np.logspace(-10, -1, num=30)
list_ridge_scores = []
for alpha in alphas:
    ridge.set_params(ridge__alpha=alpha)
    ridge.fit(X_sub_train, y_sub_train)
    list_ridge_scores.append(ridge.score(X_valid, y_valid))

# %%
plt.plot(alphas, list_ridge_scores, "+-", label='Ridge')
plt.xlabel('alpha (regularization strength)')
plt.ylabel('R2 score (higher is better)')
_ = plt.legend()

# %% [markdown]
# As we can see, regularization is just like salt in cooking:
# one must balance its amount to get the best performance.

# %%
best_alpha = alphas[np.argmax(list_ridge_scores)]
best_alpha

# %% [markdown]
# Finally, we can re-train a Ridge model on the full dataset,
# with the best value for alpha we found earlier, and check the score.

# %%
ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      Ridge(alpha=best_alpha))
ridge.fit(X_train, y_train)
test_score = ridge.score(X_test, y_test)

print(f"R2 score of ridge model on the test set:\n"
      f"{test_score:.3f}")

# %% [markdown]
# In the next exercise, you will use a scikit-learn estimator which allows to
# make some parameters tuning instead of programming yourself a `for` loop by
# hand.
#
# In this notebook, you learned about the concept of regularization and
# the importance of preprocessing and parameter tuning.
