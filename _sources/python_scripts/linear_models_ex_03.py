# %% [markdown]
# # 📝 Exercise 03
#
# In all previous notebooks, we only used a single feature in `data`. But we
# have already shown that we could add new features to make the model more
# expressive by deriving new features, based on the original feature.
#
# The aim of this notebook is to train a linear regression algorithm on a
# dataset more than a single feature.
#
# We will load a dataset about house prices in California.
# The dataset consists of 8 features regarding the demography and geography of
# districts in California and the aim is to predict the median house price of
# each district. We will use all 8 features to predict the target, the median
# house price.

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
data.head()

# %% [markdown]
# Now this is your turn to train a linear regression model on this dataset.
# You will need to:
# * create a linear regression model;
# * execute a cross-validation with 10 folds and use the mean absolute error
#   (MAE) as metric. Ensure to return the fitted estimators;
# * compute mean and std of the MAE in thousands of dollars (k$);
# * show the values of the coefficients for each feature using a boxplot by
#   inspecting the fitted model returned from the cross-validation.

# %%
# Write your code here.: make the exercise
