# %% [markdown]
# # Presentation of the datasets
#
# Before to present tree-based models, we will make a quick presentation of the
# [Palmer penguins dataset](https://allisonhorst.github.io/palmerpenguins/)
# dataset. We will use this dataset for both classification and regression
# problems by selecting a subset of the feature to make our explanations
# intuitive.
#
# ## Classification dataset
#
# We will use this dataset in classification setting to predict the penguins'
# species from anatomical information.
#
# Each penguin is from one of the three following species: Adelie, Gentoo, and
# Chinstrap. See the illustration below depicting the three different penguin
# species:
#
# ![Image of penguins](https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/lter_penguins.png)
#
# This problem is a classification problem since the target is categorical.
# We will limit our input data to a subset of the original features
# to simplify our explanations when presenting the decision tree algorithm.
# Indeed, we will use feature based on penguins' culmen measurement. You can
# learn more about the penguins' culmen with illustration below:
#
# ![Image of culmen](https://github.com/allisonhorst/palmerpenguins/raw/master/man/figures/culmen_depth.png)
#
# We will start by loading this subset of the dataset.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_classification.csv")

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

# %% [markdown]
# Let's check the dataset more into details.

# %%
data.head()

# %% [markdown]
# Since that we have few samples, we can check a scatter plot to observe the
# samples distribution.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

_, axs = plt.subplots(ncols=3, figsize=(15, 4))

sns.scatterplot(
    x=culmen_columns[0], y=culmen_columns[1], hue=target_column,
    data=data, ax=axs[0])
sns.kdeplot(
    data=data, x=culmen_columns[0], hue=target_column,
    ax=axs[1])
sns.kdeplot(
    data=data, x=culmen_columns[1], hue=target_column,
    ax=axs[2])

# %% [markdown]
# We can first check the feature distributions by looking at the diagonal plots
# of the pairplot. We can build the following intuitions:
#
# * The Adelie species is separable from the Gentoo and Chinstrap species using
#   the culmen length;
# * The Gentoo species is separable from the Adelie and Chinstrap species using
#   the culmen depth.
#
# ## Regression dataset
#
# In regression setting, the target is a continuous variable instead of
# categories. Here, we use two features of the dataset to make such a problem:
# the flipper length will be used as data and the body mass will be the target.
# In short, we want to predict the body mass using the flipper length.
#
# We will load the dataset and visualize the relationship between the flipper
# length and the body mass of penguins.

# %%
data = pd.read_csv("../datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

# %%
sns.scatterplot(data=data, x=data_columns[0], y=target_column)

# %% [markdown]
# Here, we deal with a regression problem because our target is a continuous
# variable ranging from 2.7 kg to 6.3 kg. From the scatter plot above, we can
# observe that we have a linear relationship between the flipper length
# and the body mass. The longer the flipper of a penguin, the heavier the
# penguin.
