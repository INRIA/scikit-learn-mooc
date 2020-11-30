# %% [markdown]
# # Regression
# Unlike in classification problems, the target `y` is a continuous
# variable in regression problems. Therefore, classification metrics cannot
# be used to evaluate the performance of regression models. Instead, there
# exists a set of metrics dedicated to regression.

# %%
import pandas as pd

data = pd.read_csv(
    ("https://raw.githubusercontent.com/christophM/interpretable-ml-book/"
     "master/data/bike.csv"),
)
# rename the columns with human-readable names
data = data.rename(columns={
    "yr": "year", "mnth": "month", "temp": "temperature", "hum": "humidity",
    "cnt": "count", "days_since_2011": "days since 2011"
})
# convert the categorical columns with a proper category data type
for col in data.columns:
    if data[col].dtype.kind == "O":
        data[col] = data[col].astype("category")

# separate the target from the original data
X = data.drop(columns=["count"])
y = data["count"]

# %%
X.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

plt.hist(y, bins=50, density=True)
plt.xlabel("Number of bike rentals")
plt.ylabel("Probability")
_ = plt.title("Target distribution")

# %% [markdown]
# Our problem can be formulated as follows: we would like to infer the number
# of bike rentals in a day using information about the day. The number of bike
# rentals is a number that can vary in the interval [0, max_number_of_bikes).
# As in the previous section, we will train a
# model and evaluate its performance while introducing different
# regression metrics.
#
# First, we split the data into training and a testing sets.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=0
)

# %% [markdown]
# ### Baseline model
# We will use a random forest as a model. However, we first need to check the
# type of data that we are dealing with:

# %%
X_train.info()

# %% [markdown]
# While some features are numeric, some have been tagged as `category`. These
# features need to be encoded such that our random forest can
# deal with them. The simplest solution is to use an `OrdinalEncoder`.
# Regarding, the numerical features, we don't need to do anything. Thus, we
# will create preprocessing steps to take care of the encoding.

# %%
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_selector = selector(dtype_include="category")
preprocessor = make_column_transformer(
    (OrdinalEncoder(), categorical_selector),
    remainder="passthrough",
)

X_train_preprocessed = pd.DataFrame(
    preprocessor.fit_transform(X_train),
    columns=(
        categorical_selector(X_train) +
        [col for col in X_train.columns
         if col not in categorical_selector(X_train)]
    )
)
X_train_preprocessed.head()

# %% [markdown]
# Just to have some insight about the preprocessing, we preprocess
# the training data show the result. We can observe that the original strings
# are now encoded with numbers. We can now create our model.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

regressor = make_pipeline(preprocessor, RandomForestRegressor())
regressor.fit(X_train, y_train)

# %% [markdown]
# As for scikit-learn classifiers, scikit-learn regressors have a `score`
# method that computes the
# :math:`R^2` score (also known as the coefficient of determination):

# %%
regressor.score(X_test, y_test)

# %% [markdown]
# The :math:`R^2` score represents the proportion of variance of the target
# that is explained by the independent variables in the model. The best score
# possible
# is 1 but there is no lower bound. However, a model that predicts the
# expected value of the target would get a score of 0.

# %%
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train).score(X_test, y_test)

# %% [markdown]
# The :math:`R^2` score gives insight into the goodness of fit of the
# model. However, this score cannot be compared from one dataset to another and
# the value obtained does not have a meaningful interpretation relative the
# original unit of the target. If we wanted to get an interpretable score, we
# would be interested in the median or mean absolute error.

# %%
from sklearn.metrics import mean_absolute_error

y_pred = regressor.predict(X_test)
print(
    f"Mean absolute error: {mean_absolute_error(y_test, y_pred):.0f}"
)

# %% [markdown]
# By computing the mean absolute error, we can interpret that our model is
# predicting on average 507 bike rentals away from the truth. A disadvantage
# of this metric is that the mean can be
# impacted by large error. For some applications, we might not want these
# large errors to have such a big influence on our metric. In this case we can
# use the median absolute error.

# %%
from sklearn.metrics import median_absolute_error

print(
    f"Median absolute error: {median_absolute_error(y_test, y_pred):.0f}"
)

# %% [markdown]
# This metric tells us that, our model makes a median error of 405 bikes.
# FIXME: **not sure how to introduce the `mean_squared_error`.**

# %% [markdown]
# In addition of metrics, we can visually represent the results by plotting
# the predicted values versus the true values.


# %%
import numpy as np


def plot_predicted_vs_actual(y_true, y_pred, title=None):
    plt.scatter(y_true, y_pred)

    max_value = np.max([y_true.max(), y_pred.max()])
    plt.plot(
        [0, max_value],
        [0, max_value],
        color="tab:orange",
        linewidth=3,
        label="Perfect fit",
    )

    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.axis("square")
    plt.legend()
    if title is not None:
        plt.title(title)


plot_predicted_vs_actual(y_test, y_pred)

# %% [markdown]
# On this plot, correct predictions would lie on the diagonal line. This plot
# allows us to detect if the model makes errors in a consistent way, i.e.
# has some bias.
#
# Let's take an example using the house prices in Ames.

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

data = pd.read_csv("../datasets/house_prices.csv")
X, y = data.drop(columns="SalePrice"), data["SalePrice"]
X = X.select_dtypes(np.number)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# %% [markdown]
# We will fit a ridge regressor on the data and plot the prediction versus the
# actual values.

# %%
model = make_pipeline(StandardScaler(), RidgeCV())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plot_predicted_vs_actual(y_test, y_pred, title="House prices in Ames")

# %% [markdown]
# On this plot, we see that for the large True price values, our model tends to
# under-estimate the price of the house. Typically, this issue arises when
# the target to predict does not follow a normal distribution. In these cases
# the model would benefit from target transformation.

# %%
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

model_transformed_target = TransformedTargetRegressor(
    regressor=model,
    transformer=QuantileTransformer(
        n_quantiles=900, output_distribution="normal"
    ),
)
model_transformed_target.fit(X_train, y_train)
y_pred = model_transformed_target.predict(X_test)

plot_predicted_vs_actual(y_test, y_pred, title="House prices in Ames")

# %% [markdown]
# Thus, once we transformed the target, we see that we corrected some of the
# high values.
