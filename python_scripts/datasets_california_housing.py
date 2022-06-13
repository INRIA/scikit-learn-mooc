# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # The California housing dataset
#
# In this notebook, we will quickly present the dataset known as the
# "California housing dataset". This dataset can be fetched from internet using
# scikit-learn.

# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

# %% [markdown]
# We can have a first look at the available description

# %%
print(california_housing.DESCR)

# %% [markdown]
# Let's have an overview of the entire dataset.

# %%
california_housing.frame.head()

# %% [markdown]
# As written in the description, the dataset contains aggregated data regarding
# each district in California. Let's have a close look at the features that can
# be used by a predictive model.

# %%
california_housing.data.head()

# %% [markdown]
# In this dataset, we have information regarding the demography (income,
# population, house occupancy) in the districts, the location of the districts
# (latitude, longitude), and general information regarding the house in the
# districts (number of rooms, number of bedrooms, age of the house). Since
# these statistics are at the granularity of the district, they corresponds to
# averages or medians.
#
# Now, let's have a look to the target to be predicted.

# %%
california_housing.target.head()

# %% [markdown]
# The target contains the median of the house value for each district.
# Therefore, this problem is a regression problem.
#
# We can now check more into details the data types and if the dataset contains
# any missing value.

# %%
california_housing.frame.info()

# %% [markdown]
# We can see that:
#
# * the dataset contains 20,640 samples and 8 features;
# * all features are numerical features encoded as floating number;
# * there is no missing values.
#
# Let's have a quick look at the distribution of these features by plotting
# their histograms.

# %%
import matplotlib.pyplot as plt

california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)


# %% [markdown]
# We can first focus on features for which their distributions would be more or
# less expected.
#
# The median income is a distribution with a long tail. It means that the
# salary of people is more or less normally distributed but there is some
# people getting a high salary.
#
# Regarding the average house age, the distribution is more or less uniform.
#
# The target distribution has a long tail as well. In addition, we have a
# threshold-effect for high-valued houses: all houses with a price above 5 are
# given the value 5.
#
# Focusing on the average rooms, average bedrooms, average occupation, and
# population, the range of the data is large with unnoticeable bin for the
# largest values. It means that there are very high and few values (maybe they
# could be considered as outliers?). We can see this specificity looking at the
# statistics for these features:

# %%
features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
california_housing.frame[features_of_interest].describe()

# %% [markdown]
# For each of these features, comparing the `max` and `75%` values, we can see
# a huge difference. It confirms the intuitions that there are a couple of
# extreme values.
#
# Up to know, we discarded the longitude and latitude that carry geographical
# information. In short, the combination of this feature could help us to
# decide if there are locations associated with high-valued houses. Indeed,
# we could make a scatter plot where the x- and y-axis would be the latitude
# and longitude and the circle size and color would be linked with the house
# value in the district.

# %%
import seaborn as sns

sns.scatterplot(data=california_housing.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95),
           loc="upper left")
_ = plt.title("Median house value depending of\n their spatial location")

# %% [markdown]
# If you are not familiar with the state of California, it is interesting to
# notice that all datapoints show a graphical representation of this state.
# We note that the high-valued houses will be located on the coast, where the
# big cities from California are located: San Diego, Los Angeles, San Jose, or
# San Francisco.
#
# We can do a random subsampling to have less data points to plot but that
# could still allow us to see these specificities.

# %%
import numpy as np

rng = np.random.RandomState(0)
indices = rng.choice(np.arange(california_housing.frame.shape[0]), size=500,
                     replace=False)

# %%
sns.scatterplot(data=california_housing.frame.iloc[indices],
                x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 1),
           loc="upper left")
_ = plt.title("Median house value depending of\n their spatial location")

# %% [markdown]
# We can make a final analysis by making a pair plot of all features and the
# target but dropping the longitude and latitude. We will quantize the target
# such that we can create proper histogram.

# %%
import pandas as pd

# Drop the unwanted columns
columns_drop = ["Longitude", "Latitude"]
subset = california_housing.frame.iloc[indices].drop(columns=columns_drop)
# Quantize the target and keep the midpoint for each interval
subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)
subset["MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)

# %%
_ = sns.pairplot(data=subset, hue="MedHouseVal", palette="viridis")

# %% [markdown]
# While it is always complicated to interpret a pairplot since there is a lot
# of data, here we can get a couple of intuitions. We can confirm that some
# features have extreme values (outliers?). We can as well see that the median
# income is helpful to distinguish high-valued from low-valued houses.
#
# Thus, creating a predictive model, we could expect the longitude, latitude,
# and the median income to be useful features to help at predicting the median
# house values.
#
# If you are curious, we created a linear predictive model below and show the
# values of the coefficients obtained via cross-validation

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

alphas = np.logspace(-3, 1, num=30)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
cv_results = cross_validate(
    model, california_housing.data, california_housing.target,
    return_estimator=True, n_jobs=2)

# %%
score = cv_results["test_score"]
print(f"R2 score: {score.mean():.3f} ± {score.std():.3f}")

# %%
import pandas as pd

coefs = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns=california_housing.feature_names
)

# %%
color = {"whiskers": "black", "medians": "black", "caps": "black"}
coefs.plot.box(vert=False, color=color)
plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
_ = plt.title("Coefficients of Ridge models\n via cross-validation")

# %% [markdown]
# It seems that the three features that we earlier spotted are found important
# by this model. But be careful regarding interpreting these coefficients.
# We let you go into the module "Interpretation" to go in depth regarding such
# experiment.
