# %% [markdown]
# # Regression
#
# In this notebook, we will present the metrics that can be used in regression.
#
# Unlike in classification problems, the target `y` is a continuous variable in
# regression problems. Therefore, classification metrics cannot be used to
# evaluate the performance of regression models. Instead, there exists a set of
# metrics dedicated to regression.
#
# We will use the Ames housing dataset where the goal is to predict the price
# of houses in Ames town. As for the classification, we will only use a single
# train-test split to focus only on the regression metrics.

# %%
import pandas as pd
import numpy as np

data = pd.read_csv("../datasets/house_prices.csv")
X, y = data.drop(columns="SalePrice"), data["SalePrice"]
X = X.select_dtypes(np.number)
y /= 1000

# %% [markdown]
# Let's start by splitting our dataset intro a train and test set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0
)

# %% [markdown]
# Some machine learning models were designed to be solved as an optimization
# problem: minimzing an error (also known as loss function) using a training
# set. A basic loss function used in regression is the mean squared error.
# Thus, this metric is sometimes used to evaluate a model since this is also
# the loss function optimized by a model.
#
# We will give an example using a linear regression model.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)

print(f"Mean squared error on the training set: "
      f"{mean_squared_error(y_train, y_pred):.3f}")

# %% [markdown]
# Our linear regression model is the moodel minimizing the mean squared error
# on the training set. It means that there is no other set of coefficients
# which will decrease the error.
#
# Then, we can compute the mean squared error on the test set.

# %%
y_pred = regressor.predict(X_test)

print(f"Mean squared error on the testing set: "
      f"{mean_squared_error(y_test, y_pred):.3f}")

# %% [markdown]
# The raw MSE can be difficult to interpret. One way is to rescale the MSE
# by the variance of the target. This score is known as the $R^2$ also called
# the coefficient of determination. Indeed, this is the default score used
# in scikit-learn by calliing the method `score`.

# %%
regressor.score(X_test, y_test)

# %% [markdown]
# The $R^2$ score represents the proportion of variance of the target that is
# explained by the independent variables in the model. The best score possible
# is 1 but there is no lower bound. However, a model that predicts the expected
# value of the target would get a score of 0.

# %%
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train)
print(f"R2 score for a regressor predicting the mean:"
      f"{dummy_regressor.score(X_test, y_test):.3f}")

# %% [markdown]
# The $R^2$ score gives insight into the goodness of fit of the model. However,
# this score cannot be compared from one dataset to another and the value
# obtained does not have a meaningful interpretation relative the original unit
# of the target. If we wanted to get an interpretable score, we would be
# interested in the median or mean absolute error.

# %%
from sklearn.metrics import mean_absolute_error

y_pred = regressor.predict(X_test)
print(f"Mean absolute error: "
      f"{mean_absolute_error(y_test, y_pred):.3f} k$")

# %% [markdown]
# By computing the mean absolute error, we can interpret that our model is
# predicting on average 22.6 k$ away from the true house price. A disadvantage
# of this metric is that the mean can be impacted by large error. For some
# applications, we might not want these large errors to have such a big
# influence on our metric. In this case we can use the median absolute error.

# %%
from sklearn.metrics import median_absolute_error

print(f"Median absolute error: "
      f"{median_absolute_error(y_test, y_pred):.3f} k$")

# %% [markdown]
# **FIXME: in 0.24, introduce median absolute percentage error**

# %% [markdown]
# In addition of metrics, we can visually represent the results by plotting
# the predicted values versus the true values.

# %%
predicted_actual = {
    "True values (k$)": y_test, "Predicted values (k$)": y_pred}
predicted_actual = pd.DataFrame(predicted_actual)

# %%
import seaborn as sns
sns.set_context("talk")

ax = sns.scatterplot(
    data=predicted_actual, x="True values (k$)", y="Predicted values (k$)")
ax.axline((0, 0), slope=1, color="tab:orange", label="Perfect fit")
ax.set_aspect('equal', 'box')
_ = ax.legend()

# %% [markdown]
# On this plot, correct predictions would lie on the diagonal line. This plot
# allows us to detect if the model makes errors in a consistent way, i.e.
# has some bias.
#
# On this plot, we see that for the large True price values, our model tends to
# under-estimate the price of the house. Typically, this issue arises when the
# target to predict does not follow a normal distribution. In these cases the
# model would benefit from target transformation.

# %%
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

transformer = QuantileTransformer(
    n_quantiles=900, output_distribution="normal")
model_transformed_target = TransformedTargetRegressor(
    regressor=regressor, transformer=transformer)
model_transformed_target.fit(X_train, y_train)
y_pred = model_transformed_target.predict(X_test)

# %%
predicted_actual = {
    "True values (k$)": y_test, "Predicted values (k$)": y_pred}
predicted_actual = pd.DataFrame(predicted_actual)

ax = sns.scatterplot(
    data=predicted_actual, x="True values (k$)", y="Predicted values (k$)")
ax.axline((0, 0), slope=1, color="tab:orange", label="Perfect fit")
ax.set_aspect('equal', 'box')
_ = ax.legend()

# %% [markdown]
# Thus, once we transformed the target, we see that we corrected some of the
# high values.
