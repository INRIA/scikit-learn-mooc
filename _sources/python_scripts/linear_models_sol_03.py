# %% [markdown]
# # 📃 Solution for Exercise 03
#
# In all previous notebooks, we only used a single feature in `X`. But we have
# already shown that we could add new feature to make the model more expressive
# by deriving new features, based on the original feature.
#
# The aim of this notebook is to train a linear regression algorithm on a
# dataset more than a single feature.
#
# We will load a dataset about house prices in California.
# The dataset consists of 8 features regarding the demography and geography of
# districts in California and the aim is to predict the median house price of
# each district. We will use all 8 features to predict the target, median
# house price.

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X.head()

# %% [markdown]
# Now this is your turn to train a linear regression model on this dataset.
# You will need to:
# * split the dataset into a training and testing set;
# * train a linear regression model on the training set;
# * compute the mean absolute error in k$;
# * show the values of the coefficients for each feature.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_absolute_error

y_pred = linear_regression.predict(X_test)
print(f"Mean absolute error on testing set: "
      f"{mean_absolute_error(y_test, y_pred):.3f} k$")

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

weights = pd.Series(linear_regression.coef_, index=X.columns)
weights.plot(kind="barh")
_ = plt.title("Value of linear regression coefficients")

# %%
