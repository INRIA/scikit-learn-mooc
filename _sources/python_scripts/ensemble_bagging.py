# %% [markdown]
# # Bagging
#
# In this notebook, we will present the first ensemble using bootstrap samples
# called bagging.
#
# Bagging stands for Bootstrap AGGregatING. It uses bootstrap (random sampling
# with replacement) to learn several models. At predict time, the predictions
# of each learner are aggregated to give the final predictions.
#
# First, we will generate a simple synthetic dataset to get insights regarding
# bootstraping.

# %%
import pandas as pd
import numpy as np

# create a random number generator that will be used to set the randomness
rng = np.random.RandomState(0)


def generate_data(n_samples=50):
    """Generate synthetic dataset. Returns `data_train`, `data_test`,
    `target_train`."""
    x_max, x_min = 1.4, -1.4
    len_x = x_max - x_min
    x = rng.rand(n_samples) * len_x - len_x / 2
    noise = rng.randn(n_samples) * 0.3
    y = x ** 3 - 0.5 * x ** 2 + noise

    data_train = pd.DataFrame(x, columns=["Feature"])
    data_test = pd.DataFrame(
        np.linspace(x_max, x_min, num=300), columns=["Feature"])
    target_train = pd.Series(y, name="Target")

    return data_train, data_test, target_train


# %%
import matplotlib.pyplot as plt
import seaborn as sns

data_train, data_test, target_train = generate_data(n_samples=50)
sns.scatterplot(x=data_train["Feature"], y=target_train, color="black",
                alpha=0.5)
_ = plt.title("Synthetic regression dataset")

# %% [markdown]
# The link between our feature and the target to predict is non-linear.
# However, a decision tree is capable of fitting such data.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3, random_state=0)
tree.fit(data_train, target_train)
y_pred = tree.predict(data_test)

# %%
sns.scatterplot(x=data_train["Feature"], y=target_train, color="black",
                alpha=0.5)
plt.plot(data_test, y_pred, label="Fitted tree")
plt.legend()
_ = plt.title("Predictions by a single decision tree")

# %% [markdown]
# Let's see how we can use bootstraping to learn several trees.
#
# ## Bootstrap sample
#
# A bootstrap sample corresponds to a resampling, with replacement, of the
# original dataset, a sample that is the same size as the original dataset.
# Thus, the bootstrap sample will contain some data points several times while
# some of the original data points will not be present.
#
# We will create a function that given `data` and `target` will return a
# bootstrap sample `data_bootstrap` and `target_bootstrap`.


# %%
def bootstrap_sample(data, target):
    # indices corresponding to a sampling with replacement of the same sample
    # size than the original data
    bootstrap_indices = rng.choice(
        np.arange(target.shape[0]), size=target.shape[0], replace=True,
    )
    data_bootstrap_sample = data.iloc[bootstrap_indices]
    target_bootstrap_sample = target.iloc[bootstrap_indices]
    return data_bootstrap_sample, target_bootstrap_sample


# %% [markdown]
# We will generate 3 bootstrap samples and qualitatively check the difference
# with the original dataset.

# %%
bootstraps_illustration = pd.DataFrame()
bootstraps_illustration["Original"] = data_train["Feature"]

n_bootstrap = 3
for bootstrap_idx in range(n_bootstrap):
    # draw a bootstrap from the original data
    bootstrap_data, target_data = bootstrap_sample(data_train, target_train)
    # store only the bootstrap sample
    bootstraps_illustration[f"Boostrap sample #{bootstrap_idx + 1}"] = \
        bootstrap_data["Feature"].to_numpy()

# %% [markdown]
# In the cell above, we generated three bootstrap samples and we stored only
# the feature values. In this manner, we will plot the features value from each
# set and check the how different they are.
#
# ```{note}
# In the next cell, we transform the dataframe from wide to long format. The
# column name become a by row information. `pd.melt` is in charge of doing this
# transformation. We make this transformation because we will use the seaborn
# function `sns.swarmplot` that expect long format dataframe.
# ```

# %%
bootstraps_illustration = bootstraps_illustration.melt(
    var_name="Type of data", value_name="Feature")

# %%
sns.swarmplot(x=bootstraps_illustration["Feature"],
              y=bootstraps_illustration["Type of data"])
_ = plt.title("Feature values for the different sets")

# %% [markdown]
# We observe that the 3 generated bootstrap samples are all different from the
# original dataset. The sampling with replacement is the cause of this
# fluctuation. To confirm this intuition, we can check the number of unique
# samples in the bootstrap samples.

# %%
data_train_huge, data_test_huge, target_train_huge = generate_data(
    n_samples=100_000)
data_bootstrap_sample, target_bootstrap_sample = bootstrap_sample(
    data_train_huge, target_train_huge)

ratio_unique_sample = (np.unique(data_bootstrap_sample).size /
                       data_bootstrap_sample.size)
print(
    f"Percentage of samples present in the original dataset: "
    f"{ratio_unique_sample * 100:.1f}%"
)

# %% [markdown]
# On average, ~63.2% of the original data points of the original dataset will
# be present in the bootstrap sample. The other ~36.8% are just repeated
# samples.
#
# So, we are able to generate many datasets, all slightly different. Now, we
# can fit a decision tree for each of these datasets and they all
# shall be slightly different as well.

# %%
forest = []
for bootstrap_idx in range(n_bootstrap):
    tree = DecisionTreeRegressor(max_depth=3, random_state=0)

    data_bootstrap_sample, target_bootstrap_sample = bootstrap_sample(
        data_train, target_train)
    tree.fit(data_bootstrap_sample, target_bootstrap_sample)
    forest.append(tree)

# %% [markdown]
# Now that we created a forest with many different trees, we can use each of
# the tree to predict on the testing data. They shall give slightly different
# results.

# %%
sns.scatterplot(x=data_train["Feature"], y=target_train, color="black",
                alpha=0.5)
for tree_idx, tree in enumerate(forest):
    target_predicted = tree.predict(data_test)
    plt.plot(data_test, target_predicted, linestyle="--", alpha=0.8,
             label=f"Tree #{tree_idx} predictions")

plt.legend()
_ = plt.title("Predictions of trees trained on different bootstraps")

# %% [markdown]
# ## Aggregating
#
# Once our trees are fitted and we are able to get predictions for each of
# them, we need to combine them. In regression, the most straightforward
# approach is to average the different predictions from all learners. We can
# plot the averaged predictions from the previous example.

# %%
sns.scatterplot(x=data_train["Feature"], y=target_train, color="black",
                alpha=0.5)

target_predicted_forest = []
for tree_idx, tree in enumerate(forest):
    target_predicted = tree.predict(data_test)
    plt.plot(data_test, target_predicted, linestyle="--", alpha=0.8,
             label=f"Tree #{tree_idx} predictions")
    target_predicted_forest.append(target_predicted)

target_predicted_forest = np.mean(target_predicted_forest, axis=0)
plt.plot(data_test, target_predicted_forest, label="Averaged predictions",
         linestyle="-")
plt.legend()
plt.title("Predictions of individual and combined tree")

# %% [markdown]
# The unbroken red line shows the averaged predictions, which would be the
# final preditions given by our 'bag' of decision tree regressors.
#
# ## Bagging in scikit-learn
#
# Scikit-learn implements bagging estimators. It takes a base model that is the
# model trained on each bootstrap sample.

# %%
from sklearn.ensemble import BaggingRegressor

bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                           n_estimators=3)
bagging.fit(data_train, target_train)
target_predicted_forest = bagging.predict(data_test)

# %%
sns.scatterplot(x=data_train["Feature"], y=target_train, color="black",
                alpha=0.5)
plt.plot(data_test, target_predicted_forest, label="Bag of decision trees")
plt.legend()
_ = plt.title("Predictions from a bagging classifier")

# %% [markdown]
# While we used a decision tree as a base model, nothing prevent us of using
# any other type of model. We will give an example that will use a linear
# regression.

# %%
from sklearn.linear_model import LinearRegression

bagging = BaggingRegressor(base_estimator=LinearRegression(),
                           n_estimators=3)
bagging.fit(data_train, target_train)
target_predicted_linear = bagging.predict(data_test)

# %%
sns.scatterplot(x=data_train["Feature"], y=target_train, color="black",
                alpha=0.5)
plt.plot(data_test, target_predicted_forest, label="Bag of decision trees")
plt.plot(data_test, target_predicted_linear, label="Bag of linear regression")
plt.legend()
_ = plt.title("Bagging classifiers using \ndecision trees and linear models")

# %% [markdown]
# However, we see that using a bag of linear models is not helpful here because
# we still obtain a linear model.
