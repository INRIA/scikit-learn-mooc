# %% [markdown]
# # 📝 Introductory exercise for non i.i.d. data
#
# ```{note}
# i.i.d is the acronym of "independent and identically distributed"
# (as in "independent and identically distributed random variables").
# ```
#
# This exercise aims at showing some aspects to consider when dealing with non
# i.i.d data, typically time series.
#
# For this purpose, we will create a synthetic dataset simulating stock values.
# We will formulate the following data science problem: predict the value of a
# specific stock given other stock.
#
# To make this problem interesting, we want to ensure that any predictive model
# should **not** work. In this regard, the stocks will be generated completely
# randomly without any link between them. We will only add a constraint: the
# value of a stock at a given time `t` will depend on the value of the stock
# from the past.
#
# We will create a function to generate such data.

# %%
import numpy as np
import pandas as pd


def generate_random_stock_market(n_stock=3, seed=0):
    rng = np.random.RandomState(seed)

    date_range = pd.date_range(start="01/01/2010", end="31/12/2020")
    stocks = np.array([
        rng.randint(low=100, high=200) +
        np.cumsum(rng.normal(size=(len(date_range),)))
        for _ in range(n_stock)
    ]).T
    return pd.DataFrame(
        stocks,
        columns=[f"Stock {i}" for i in range(n_stock)],
        index=date_range,
    )


# %% [markdown]
# Now that we have our data generator, let's create three quotes, corresponding
# to the quotes of three different companies for instance. We will plot
# the stock values

# %%
stocks = generate_random_stock_market()
stocks.head()

# %%
import matplotlib.pyplot as plt

stocks.plot()
plt.ylabel("Stock value")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = plt.title("Stock values over time")

# %% [markdown]
# Because the stocks are generated randomly, it is not possible for a
# predictive model to be able to predict the value of a stock depending on the
# other stocks. By using the cross-validation framework from the previous
# exercise, we will check that we get such expected results.
#
# First, let's organise our data into a matrix `data` and a vector `target`.
# Split the data such that the `Stock 0` is the stock that we want to predict
# and the `Stock 1` and `Stock 2` are stocks used to build our model.

# %%
# Write your code here.

# %% [markdown]
# Let's create a machine learning model. We can use a
# `GradientBoostingRegressor`.

# %%
# Write your code here.

# %% [markdown]
# Now, we have to define a cross-validation strategy to evaluate our model.
# Use a `ShuffleSplit` cross-validation.

# %%
# Write your code here.

# %% [markdown]
# We should be set to make our evaluation. Call the function `cross_val_score`
# to compute the $R^2$ score for the different split and report the mean
# and standard deviation of the model.

# %%
# Write your code here.

# %% [markdown]
# Your model is not giving random predictions. Could you ellaborate on what
# are the reasons of such a success on random data.
