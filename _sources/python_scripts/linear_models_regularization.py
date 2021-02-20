# %% [markdown]
# # Regularization of linear regression model
#
# In this notebook, we will see the limitations of linear regression models and
# the advantage of using regularized models instead.
#
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
# In one of the previous notebook, we showed that linear models could be used
# even in settings where `X` and `y` are not linearly linked.
#
# We showed that one can use the `PolynomialFeatures` transformer to create
# additional features encoding non-linear interactions between features.
#
# Here, we will use this transformer to augment the feature space.
# Subsequently, we will train a linear regression model.
# We will use the out-of-sample test set to evaluate the
# generalization capabilities of our model.

# %%
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

linear_regression = make_pipeline(PolynomialFeatures(degree=2),
                                  LinearRegression())
cv_results = cross_validate(linear_regression, X, y, cv=10,
                            return_train_score=True,
                            return_estimator=True)
test_score = cv_results["test_score"]
print(f"R2 score of linear regresion model on the test set:\n"
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %% [markdown]
# We see that we obtain an $R^2$ score below zero.
#
# It means that our model is far worse at predicting the mean of `y_train`.
# This issue is due to overfitting.
# We can compute the score on the training set to confirm this intuition.

# %%
train_score = cv_results["train_score"]
print(f"R2 score of linear regresion model on the train set:\n"
      f"{train_score.mean():.3f} +/- {train_score.std():.3f}")

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

# Define the style in the boxplot
boxplot_property = {
    "vert": False, "whis": 100, "patch_artist": True, "widths": 0.3,
    "boxprops": dict(linewidth=3, color='black', alpha=0.9),
    "medianprops": dict(linewidth=2.5, color='black', alpha=0.9),
    "whiskerprops": dict(linewidth=3, color='black', alpha=0.9),
    "capprops": dict(linewidth=3, color='black', alpha=0.9),
}

# %%
weights_linear_regression = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns=cv_results["estimator"][0][0].get_feature_names(
        input_features=X.columns))
_, ax = plt.subplots(figsize=(6, 16))
weights_linear_regression.plot.box(ax=ax, **boxplot_property)
ax.set_title("Linear regression coefficients")

# %% [markdown]
# We can force the linear regression model to consider all features in a more
# homogeneous manner. In fact, we could force large positive or negative weight
# to shrink toward zero. This is known as regularization. We will use a ridge
# model which enforces such behavior.

# %%
from sklearn.linear_model import Ridge

ridge = make_pipeline(PolynomialFeatures(degree=2),
                      Ridge(alpha=100))
cv_results = cross_validate(ridge, X, y, cv=10,
                            return_train_score=True,
                            return_estimator=True)
test_score = cv_results["test_score"]
print(f"R2 score of ridge model on the test set:\n"
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %%
train_score = cv_results["train_score"]
print(f"R2 score of ridge model on the train set:\n"
      f"{train_score.mean():.3f} +/- {train_score.std():.3f}")

# %% [markdown]
# We see that the training and testing scores are much closer, indicating that
# our model is less overfitting. We can compare the values of the weights of
# ridge with the un-regularized linear regression.

# %%
weights_ridge = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns=cv_results["estimator"][0][0].get_feature_names(
        input_features=X.columns))

# %%
_, axs = plt.subplots(ncols=2, figsize=(12, 16))
weights_linear_regression.plot.box(ax=axs[0], **boxplot_property)
weights_ridge.plot.box(ax=axs[1], **boxplot_property)
axs[1].set_yticklabels([""] * len(weights_ridge.columns))
axs[0].set_title("Linear regression weights")
_ = axs[1].set_title("Ridge weights")

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
#
# Indeed, the larger alpha is, the larger this enforcement will be.
#
# This procedure should make us think about feature rescaling.
# Let's consider the case where features have an identical data dispersion:
# if two features are found equally important by the model, they will be
# affected similarly by regularization strength.
#
# Now, let's consider the scenario where features have completely different
# data dispersion (for instance age in years and annual revenue in dollars).
# If two features are as important, our model will boost the weights of
# features with small dispersion and reduce the weights of features with
# high dispersion.
#
# We recall that regularization forces weights to be closer. Therefore, we get
# an intuition that if we want to use regularization, dealing with rescaled
# data would make it easier to find an optimal regularization parameter and
# thus an adequate model.
#
# As a side note, some solvers based on gradient computation are expecting such
# rescaled data. Unscaled data will be detrimental when computing the optimal
# weights. Therefore, when working with a linear model and numerical data, it
# is generally good practice to scale the data.
#
# Thus, we will add a `StandardScaler` in the machine learning pipeline. This
# scaler will be placed just before the regressor.

# %%
from sklearn.preprocessing import StandardScaler

ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      Ridge(alpha=0.5))
cv_results = cross_validate(ridge, X, y, cv=10,
                            return_train_score=True,
                            return_estimator=True)
test_score = cv_results["test_score"]
print(f"R2 score of ridge model on the test set:\n"
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %%
train_score = cv_results["train_score"]
print(f"R2 score of ridge model on the train set:\n"
      f"{train_score.mean():.3f} +/- {train_score.std():.3f}")

# %% [markdown]
# As we can see in this example, using a pipeline simplifies the manual
# handling.
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
# Model hyperparameters tuning should be done with care. Indeed, we want to
# find an optimal parameter that maximizes some metrics. Thus, it requires both
# a training set and testing set.
#
# However, this testing set should be different from the out-of-sample testing
# set that we used to evaluate our model: if we use the same one, we are using
# an `alpha` which was optimized for this testing set and it breaks the
# out-of-sample rule.
#
# Therefore, we should include search of the hyperparameter `alpha` within the
# cross-validation. As we saw in previous notebooks, we could use a
# grid-search. However, some predictor in scikit-learn are available with
# an integrated hyperparameter search, more efficient than using a grid-search.
# The name of these predictors finishes by `CV`. In the case of `Ridge`,
# scikit-learn provides a `RidgeCV` regressor.
#
# Therefore, we can use this predictor as the last step of the pipeline.
# Including the pipeline a cross-validation allows to make a nested
# cross-validation: the inner cross-validation will search for the best
# alpha, while the outer cross-validation will give an estimate of the
# generalization score.

# %%
import numpy as np
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-2, 0, num=20)
ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      RidgeCV(alphas=alphas, store_cv_values=True))

# %%
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, random_state=1)
cv_results = cross_validate(ridge, X, y, cv=cv,
                            return_train_score=True,
                            return_estimator=True,
                            n_jobs=-1)
test_score = cv_results["test_score"]
print(f"R2 score of ridge model with optimal alpha on the test set:\n"
      f"{test_score.mean():.3f} +/- {test_score.std():.3f}")

# %%
train_score = cv_results["train_score"]
print(f"R2 score of ridge model on the train set:\n"
      f"{train_score.mean():.3f} +/- {train_score.std():.3f}")

# %% [markdown]
# By optimizing `alpha`, we see that the training an testing scores are closed.
# It indicates that our model is not overfitting.
#
# When fitting the ridge regressor, we also requested to store the error found
# during cross-validation (by setting the parameter `store_cv_values=True`).
# We will plot the mean MSE for the different `alphas`.

# %%
cv_alphas = pd.DataFrame(
    [est[-1].cv_values_.mean(axis=0) for est in cv_results["estimator"]],
     columns=alphas)

_, ax = plt.subplots()
cv_alphas.mean(axis=0).plot(ax=ax, marker="+")
ax.set_ylabel("Mean squared error\n (lower is better)")
ax.set_xlabel("alpha")
_ = ax.set_title("Error obtained by cross-validation")

# %% [markdown]
# As we can see, regularization is just like salt in cooking: one must balance
# its amount to get the best performance. We can check if the best `alpha`
# found is stable across the cross-validation fold.

# %%
best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
best_alphas

# %% [markdown]
# In this notebook, you learned about the concept of regularization and
# the importance of preprocessing and parameter tuning.
