# %% [markdown]
# # The Ames housing dataset

# %%
import pandas as pd

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values='?')

# %%
ames_housing.head()

# %%
data = ames_housing.drop(columns=["Id", "SalePrice"])

# %%
target = ames_housing["SalePrice"]

# %%
data.info()

# %%
numerical_data = data.select_dtypes("number")

# %%
numerical_data.info()

# %%
import matplotlib.pyplot as plt

numerical_data.hist(bins=20, figsize=(12, 22), edgecolor="black", density=True,
                    layout=(9, 4))
plt.subplots_adjust(hspace=0.8, wspace=0.8)

# %%
string_data = data.select_dtypes(object)

# %%
string_data.info()

# %%
from math import ceil

n_string_features = string_data.shape[1]
nrows, ncols = ceil(n_string_features / 4), 4

fig, axs = plt.subplots(ncols=ncols, nrows=nrows)

for feature_name, ax in zip(string_data, axs.ravel()):
    string_data[feature_name].value_counts().plot.bar(ax=ax)

# %%
