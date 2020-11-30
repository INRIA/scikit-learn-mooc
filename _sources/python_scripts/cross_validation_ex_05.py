# %% [markdown]
# # Introductory exercise for non i.i.d. data
#
# This exercise aims at showing some aspects to consider when dealing non i.i.d
# data, typically time series.
#
# For this purpose, we will create a synthetic dataset simulating stocks.
# %%
import numpy as np
import pandas as pd


def generate_random_stock_market(n_stock=4, seed=0):
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


# %%
stocks = generate_random_stock_market()


# %%
stocks.plot()

# %%
from sklearn.model_selection import train_test_split

X, y = stocks.drop(columns="Stock 0"), stocks["Stock 0"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# %%
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# %%
from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
r2_score(y_test, y_pred)


# %%
