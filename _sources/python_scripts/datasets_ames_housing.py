# %% [markdown]
# # The Ames housing dataset
#
# In this notebook, we will quickly present the "Ames housing" dataset. We will
# see that this dataset is similar to the "California housing" dataset.
# However, it is more complex to handle: it contains missing data and both
# numerical and categorical features.
#
# This dataset is located in the `datasets` directory. It is stored in a comma
# separated value (CSV) file. As previously mentioned, we are aware that the
# dataset contains missing values. The character `"?"` is used as a missing
# value marker.
#
# We will open the dataset and specify the missing value marker such that they
# will be parsed by pandas when opening the file.

# %%
import pandas as pd

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values='?')

# %% [markdown]
# We can have a first look at the available columns in this dataset.

# %%
ames_housing.head()

# %% [markdown]
# We see that the last column named `"SalePrice"` is indeed the target that we
# would like to predict. So we will split our dataset into two variables
# containing the data and the target.

# %%
data = ames_housing.drop(columns=["Id", "SalePrice"])
target = ames_housing["SalePrice"]

# %% [markdown]
# Let's have a quick look at the target before to focus on the data.

# %%
target.head()

# %% [markdown]
# We see that the target contains continuous value. It corresponds to the price
# of a house in $. We can have a look at the target distribution.

# %%
import matplotlib.pyplot as plt
target.plot.hist(bins=20, edgecolor="black")
plt.xlabel("House price in $")
_ = plt.title("Distribution of the house price \nin Ames")

# %% [markdown]
# We see that the distribution has a long tail. It means that most of the house
# are normally distributed but a couple of houses have a higher than normal
# value. It could be critical to take this peculiarity into account when
# designing a predictive model.
#
# Now, we can have a look at the available data that we could use to predict
# house prices.

# %%
data.info()

# %% [markdown]
# Looking at the dataframe general information, we can see that 79 features are
# availables and that the dataset contains 1460 samples. However, some features
# contains missing values. Also, the type of data is heterogeneous: both
# numerical and categorical data are available.
#
# First, we will have a look at the data represented with numbers.

# %%
numerical_data = data.select_dtypes("number")
numerical_data.info()

# %% [markdown]
# We see that the data are mainly represented with integer number. Let's have
# a look at the histogram for all these features.

# %%
numerical_data.hist(bins=20, figsize=(12, 22), edgecolor="black", density=True,
                    layout=(9, 4))
plt.subplots_adjust(hspace=0.8, wspace=0.8)

# %% [markdown]
# We see that some features have high picks for 0. It could be linked that this
# value was assigned when the criterion did not apply, for instance the
# area of the swimming pool when no swimming pools are available.
#
# We also have some feature encoding some date (for instance year).
#
# These information are useful and should also be considered when designing a
# predictive model.
#
# Now, let's have a look at the data encoded with strings.

# %%
string_data = data.select_dtypes(object)
string_data.info()

# %% [markdown]
# These features are categorical. We can make some bar plot to see categories
# count for each feature.

# %%
from math import ceil
from itertools import zip_longest

n_string_features = string_data.shape[1]
nrows, ncols = ceil(n_string_features / 4), 4

fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(14, 80))

for feature_name, ax in zip_longest(string_data, axs.ravel()):
    if feature_name is None:
        # do not show the axis
        ax.axis("off")
        continue

    string_data[feature_name].value_counts().plot.barh(ax=ax)
    ax.set_title(feature_name)

plt.subplots_adjust(hspace=0.2, wspace=0.8)

# %% [markdown]
# Plotting this information allows us to answer to two questions:
#
# * Is there few or many categories for a given features?
# * Is there rare categories for some features?
#
# Knowing about these peculiarities would help at designing the predictive
# pipeline.
