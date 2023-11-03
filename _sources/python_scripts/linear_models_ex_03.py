# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M4.03
#
# Now, we tackle a more realistic classification problem instead of making a
# synthetic dataset. We start by loading the Adult Census dataset with the
# following snippet. For the moment we retain only the **numerical features**.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
data = adult_census.select_dtypes(["integer", "floating"])
data = data.drop(columns=["education-num"])
data

# %% [markdown]
# We confirm that all the selected features are numerical.
#
# Compute the generalization performance in terms of accuracy of a linear model
# composed of a `StandardScaler` and a `LogisticRegression`. Use a 10-fold
# cross-validation with `return_estimator=True` to be able to inspect the
# trained estimators.

# %%
# Write your code here.

# %% [markdown]
# What is the most important feature seen by the logistic regression?
#
# You can use a boxplot to compare the absolute values of the coefficients while
# also visualizing the variability induced by the cross-validation resampling.

# %%
# Write your code here.

# %% [markdown]
# Let's now work with **both numerical and categorical features**. You can
# reload the Adult Census dataset with the following snippet:

# %%
adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
data = adult_census.drop(columns=["class", "education-num"])

# %% [markdown]
# Create a predictive model where:
# - The numerical data must be scaled.
# - The categorical data must be one-hot encoded, set `min_frequency=0.01` to
#   group categories concerning less than 1% of the total samples.
# - The predictor is a `LogisticRegression`. You may need to increase the number
#   of `max_iter`, which is 100 by default.
#
# Use the same 10-fold cross-validation strategy with `return_estimator=True` as
# above to evaluate this complex pipeline.

# %%
# Write your code here.

# %% [markdown]
# By comparing the cross-validation test scores of both models fold-to-fold,
# count the number of times the model using both numerical and categorical
# features has a better test score than the model using only numerical features.

# %%
# Write your code here.

# %% [markdown]
# For the following questions, you can copy and paste the following snippet to
# get the feature names from the column transformer here named `preprocessor`.
#
# ```python
# preprocessor.fit(data)
# feature_names = (
#     preprocessor.named_transformers_["onehotencoder"].get_feature_names_out(
#         categorical_columns
#     )
# ).tolist()
# feature_names += numerical_columns
# feature_names
# ```

# %%
# Write your code here.

# %% [markdown]
# Notice that there are as many feature names as coefficients in the last step
# of your predictive pipeline.

# %% [markdown]
# Which of the following pairs of features is most impacting the predictions of
# the logistic regression classifier based on the absolute magnitude of its
# coefficients?

# %%
# Write your code here.

# %% [markdown]
# Now create a similar pipeline consisting of the same preprocessor as above,
# followed by a `PolynomialFeatures` and a logistic regression with `C=0.01`.
# Set `degree=2` and `interaction_only=True` to the feature engineering step.
# Remember not to include a "bias" feature to avoid introducing a redundancy
# with the intercept of the subsequent logistic regression.

# %%
# Write your code here.

# %% [markdown]
# By comparing the cross-validation test scores of both models fold-to-fold,
# count the number of times the model using multiplicative interactions and both
# numerical and categorical features has a better test score than the model
# without interactions.

# %%
# Write your code here.

# %%
# Write your code here.
