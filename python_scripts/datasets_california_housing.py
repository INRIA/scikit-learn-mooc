# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

# %%
california_housing.frame.head()

# %%
california_housing.data.head()

# %%
california_housing.target.head()

# %%
california_housing.frame.info()

# %%
import matplotlib.pyplot as plt

california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)

# %%
import seaborn as sns

sns.scatterplot(data=california_housing.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.8),
           loc="upper left")
_ = plt.title("Median house value depending of\n their spatial location")

# %%
import numpy as np

rng = np.random.RandomState(0)
indices = rng.choice(np.arange(california_housing.frame.shape[0]), size=1_000)

sns.scatterplot(data=california_housing.frame.iloc[indices],
                x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.8),
           loc="upper left")
_ = plt.title("Median house value depending of\n their spatial location")

# %%
