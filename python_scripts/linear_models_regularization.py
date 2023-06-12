# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Regularization of linear regression model
#
# In this notebook, we explore the limitations of linear regression models and
# demonstrate the benefits of using regularized models instead. Additionally, we
# discuss the preprocessing steps necessary when working with regularized
# models, especially when tuning the regularization parameter.
#
# We start by highlighting the problem of overfitting that can occur with a
# simple linear regression model.
#
# ## Effect of regularization
#
# We will first load the California housing dataset.

# %% [markdown]
# ```{note}
# If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.
# ```

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
data.head()

# %% [markdown]
# In one of the previous notebook, we showed that linear models could be used
# even when there is no linear relationship between the `data` and `target`.
# For instance, one can use the `PolynomialFeatures` transformer to create
# additional features that capture some non-linear interactions between them.
#
# Here, we use this transformer to augment the feature space. Subsequently, we
# train a linear regression model. We use the out-of-sample test set to evaluate
# the generalization capabilities of our model.

# %%
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

linear_regression = make_pipeline(
    PolynomialFeatures(degree=2), LinearRegression()
)
cv_results = cross_validate(
    linear_regression,
    data,
    target,
    cv=10,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
)

# %% [markdown]
# We can compare the mean squared error on the training and testing set to
# assess the generalization performance of our model.

# %%
train_error = -cv_results["train_score"]
print(
    "Mean squared error of linear regression model on the train set:\n"
    f"{train_error.mean():.2e} ± {train_error.std():.2e}"
)

# %%
test_error = -cv_results["test_score"]
print(
    "Mean squared error of linear regression model on the test set:\n"
    f"{test_error.mean():.2e} ± {test_error.std():.2e}"
)

# %% [markdown]
# The training score is much better than the testing score. Such gap between the
# training and testing scores is an indication that our model overfitted the
# training set. Indeed, this is one of the dangers when augmenting the number of
# features with a `PolynomialFeatures` transformer. One does not expect features
# such as `PoolArea * YrSold` to be predictive.
#
# We can create a dataframe to check the weights of the model: the columns
# contain the name of the features whereas the rows store the coefficients values
# of each model during the cross-validation.
#
# Since we used a `PolynomialFeatures` to augment the data, we extract the
# feature names representative of each feature combination. Scikit-learn
# provides a `feature_names_in_` method for this purpose. First, let's get the
# first fitted model from the cross-validation.

# %%
model_first_fold = cv_results["estimator"][0]

# %% [markdown]
# Now, we can access the fitted `LinearRegression` (step `-1` i.e. the last step
# of the model) to recover the feature names.

# %%
feature_names = model_first_fold[0].get_feature_names_out(
    input_features=data.columns
)
feature_names

# %% [markdown]
# The following code creates a list by iterating through the estimators and
# querying their last step for the learned `coef_`. We can then create the
# dataframe containing all the information.

# %%
import pandas as pd

coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_linear_regression = pd.DataFrame(coefs, columns=feature_names)

# %% [markdown]
# Now, let's use a box plot to see the coefficients variations.

# %%
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
weights_linear_regression.plot.box(color=color, vert=False, figsize=(6, 16))
_ = plt.title("Linear regression coefficients")

# %% [markdown]
# We can force the linear regression model to consider all features in a more
# homogeneous manner. In fact, we could force large positive or negative weights
# to shrink toward zero. This is known as regularization. We use a ridge model
# which enforces such behavior.

# %%
from sklearn.linear_model import Ridge

ridge = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=100))
cv_results = cross_validate(
    ridge,
    data,
    target,
    cv=10,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
)

# %% [markdown]
# The code cell above generates a couple of warnings because the features
# included both extremely large and extremely small values, which are causing
# numerical problems when training the predictive model. We will get to that in
# a bit.
#
# We can explore the train and test scores of this model.

# %%
train_error = -cv_results["train_score"]
print(
    "Mean squared error of ridge model on the train set:\n"
    f"{train_error.mean():.2e} ± {train_error.std():.2e}"
)

# %%
test_error = -cv_results["test_score"]
print(
    "Mean squared error of ridge model on the test set:\n"
    f"{test_error.mean():.2e} ± {test_error.std():.2e}"
)

# %% [markdown]
# We see that the training and testing scores are much closer, indicating that
# our model is less overfitting. We can compare the values of the weights of
# ridge with the un-regularized linear regression.

# %%
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_ridge = pd.DataFrame(coefs, columns=feature_names)

# %%
weights_ridge.plot.box(color=color, vert=False, figsize=(6, 16))
_ = plt.title("Ridge weights")

# %% [markdown]
# By comparing the order of magnitude of the weights on this plot with respect
# to the previous plot, we see that a ridge model enforces all weights to lay in
# a similar scale, while the overall magnitude of the weights is shrunk towards
# zero with respect to the linear regression model.
#
# However, in this example, we omitted two important aspects: (i) the need to
# scale the data and (ii) the need to search for the best regularization
# parameter.
#
# ## Feature scaling and regularization
#
# On the one hand, weights define the link between feature values and the
# predicted target. On the other hand, regularization adds constraints on the
# weights of the model through the `alpha` parameter. Therefore, the effect that
# feature rescaling has on the final weights also interacts with regularization.
#
# Let's consider the case where features live on the same scale/units: if two
# features are found to be equally important by the model, they are be affected
# similarly by regularization strength.
#
# Now, let's consider the scenario where features have completely different data
# scales (for instance age in years and annual revenue in dollars). If two
# features are as important, our model boosts the weights of features with
# small scale and reduce the weights of features with high scale.
#
# We recall that regularization forces weights to be closer. Therefore, we get
# an intuition that if we want to use regularization, dealing with rescaled data
# would make it easier to find an optimal regularization parameter and thus an
# adequate model.
#
# As a side note, some solvers based on gradient computation are expecting such
# rescaled data. Unscaled data can be detrimental when computing the optimal
# weights. Therefore, when working with a linear model and numerical data, it is
# generally good practice to scale the data.
#
# Thus, we add a `MinMaxScaler` in the machine learning pipeline, which scales
# each feature individually such that its range maps into the range between zero
# and one. We place it just before the `PolynomialFeatures` transformer as
# powers of features in the range between zero and one remain in the same range.

# %%
from sklearn.preprocessing import StandardScaler

ridge = make_pipeline(
    PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=0.5)
)
cv_results = cross_validate(
    ridge,
    data,
    target,
    cv=10,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
)

# %%
train_error = -cv_results["train_score"]
print(
    "Mean squared error of scaled ridge model on the train set:\n"
    f"{train_error.mean():.2e} ± {train_error.std():.2e}"
)

# %%
test_error = -cv_results["test_score"]
print(
    "Mean squared error of scaled ridge model on the test set:\n"
    f"{test_error.mean():.2e} ± {test_error.std():.2e}"
)

# %% [markdown]
# We observe that scaling data has a positive impact on the test score and that
# it is now closer to the train score. It means that our model is less
# overfitted and that we are getting closer to the best generalization sweet
# spot.
#
# Let's have an additional look to the different weights.

# %%
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_ridge = pd.DataFrame(coefs, columns=feature_names)

# %%
weights_ridge.plot.box(color=color, vert=False, figsize=(6, 16))
_ = plt.title("Ridge weights with data scaling")

# %% [markdown]
# Compare to the previous plots, we see that now all weight magnitudes are
# closer and that all features are more equally contributing.
#
# In the previous example, we fixed `alpha=0.5`. We will now check the impact of
# the value of `alpha` by increasing its value.

# %%
ridge = make_pipeline(
    PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=1_000_000)
)
cv_results = cross_validate(
    ridge,
    data,
    target,
    cv=10,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
)

# %%
coefs = [est[-1].coef_ for est in cv_results["estimator"]]
weights_ridge = pd.DataFrame(coefs, columns=feature_names)

# %%
weights_ridge.plot.box(color=color, vert=False, figsize=(6, 16))
_ = plt.title("Ridge weights with data scaling and large alpha")

# %% [markdown]
# Looking specifically to weights values, we observe that increasing the value
# of `alpha` decreases the weight values. A negative value of `alpha` would
# actually enhance large weights and promote overfitting.
#
# ```{note}
# Here, we only focus on numerical features. For categorical features, it is
# generally common to omit scaling when features are encoded with a
# `OneHotEncoder` since the feature values are already on a similar scale.
#
# However, this choice may depend on the scaling method and the user case. For
# instance, standard scaling categorical features that are imbalanced (e.g. more
# occurrences of a specific category) would even out the impact of
# regularization to each category. However, scaling such features in the
# presence of rare categories could be problematic (i.e. division by a very
# small standard deviation) and it can therefore introduce numerical issues.
# ```
#
# In the previous analysis, we chose the parameter beforehand and fixed it for
# the analysis. In the next section, we check how the regularization parameter
# `alpha` should be tuned.
#
# ## Tuning the regularization parameter
#
# As mentioned, the regularization parameter needs to be tuned on each dataset.
# The default parameter does not lead to the optimal model. Therefore, we need
# to tune the `alpha` parameter.
#
# Model hyperparameter tuning should be done with care. Indeed, we want to find
# an optimal parameter that maximizes some metrics. Thus, it requires both a
# training set and testing set.
#
# However, this testing set should be different from the out-of-sample testing
# set that we used to evaluate our model: if we use the same one, we are using
# an `alpha` which was optimized for this testing set and it breaks the
# out-of-sample rule.
#
# Therefore, we should include search of the hyperparameter `alpha` within the
# cross-validation. As we saw in previous notebooks, we could use a grid-search.
# However, some predictor in scikit-learn are available with an integrated
# hyperparameter search, more efficient than using a grid-search. The name of
# these predictors finishes by `CV`. In the case of `Ridge`, scikit-learn
# provides a `RidgeCV` regressor.
#
# Therefore, we can use this predictor as the last step of the pipeline.
# Including the pipeline a cross-validation allows to make a nested
# cross-validation: the inner cross-validation searches for the best alpha,
# while the outer cross-validation gives an estimate of the testing score.

# %%
import numpy as np
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-2, 0, num=21)
ridge = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    RidgeCV(alphas=alphas, store_cv_values=True),
)

# %%
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, random_state=1)
cv_results = cross_validate(
    ridge,
    data,
    target,
    cv=cv,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
    n_jobs=2,
)

# %%
train_error = -cv_results["train_score"]
print(
    "Mean squared error of tuned ridge model on the train set:\n"
    f"{train_error.mean():.2e} ± {train_error.std():.2e}"
)

# %%
test_error = -cv_results["test_score"]
print(
    "Mean squared error of tuned ridge model on the test set:\n"
    f"{test_error.mean():.2e} ± {test_error.std():.2e}"
)

# %% [markdown]
# By optimizing `alpha`, we see that the training and testing scores are close.
# It indicates that our model is not overfitting.
#
# When fitting the ridge regressor, we also requested to store the error found
# during cross-validation (by setting the parameter `store_cv_values=True`). We
# can plot the mean squared error for the different `alphas` regularization
# strengths that we tried. The error bars represent one standard deviation of the
# average mean square error across folds for a given value of `alpha`.

# %%
mse_alphas = [
    est[-1].cv_values_.mean(axis=0) for est in cv_results["estimator"]
]
cv_alphas = pd.DataFrame(mse_alphas, columns=alphas)
cv_alphas = cv_alphas.aggregate(["mean", "std"]).T
cv_alphas

# %%
plt.errorbar(cv_alphas.index, cv_alphas["mean"], yerr=cv_alphas["std"])
plt.xlim((0.0, 1.0))
plt.ylim((4_500, 11_000))
plt.ylabel("Mean squared error\n (lower is better)")
plt.xlabel("alpha")
_ = plt.title("Testing error obtained by cross-validation")

# %% [markdown]
# As we can see, regularization is just like salt in cooking: one must balance
# its amount to get the best generalization performance. We can check if the
# best `alpha` found is stable across the cross-validation fold.

# %%
best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
best_alphas

# %% [markdown]
# The optimal regularization strength is not necessarily the same on all
# cross-validation iterations. But since we expect each cross-validation
# resampling to stem from the same data distribution, it is common practice to
# choose the best `alpha` to put into production as lying in the range defined
# by:

# %%
print(
    f"Min optimal alpha: {np.min(best_alphas):.2f} and "
    f"Max optimal alpha: {np.max(best_alphas):.2f}"
)

# %% [markdown]
# This range can be reduced depending on the feature engineering and
# preprocessing.
#
# In this notebook, you learned about the concept of regularization and the
# importance of preprocessing and parameter tuning.
