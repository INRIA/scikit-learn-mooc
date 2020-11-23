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

# create a random number generator that
# will be used to set the randomness
rng = np.random.RandomState(0)


def generate_data(n_samples=50):
    """Generate synthetic dataset. Returns `X_train`, `X_test`, `y_train`."""
    x_max, x_min = 1.4, -1.4
    len_x = x_max - x_min
    x = rng.rand(n_samples) * len_x - len_x / 2
    noise = rng.randn(n_samples) * 0.3
    y = x ** 3 - 0.5 * x ** 2 + noise

    X_train = pd.DataFrame(x, columns=["Feature"])
    X_test = pd.DataFrame(
        np.linspace(x_max, x_min, num=300), columns=["Feature"])
    y_train = pd.Series(y, name="Target")

    return X_train, X_test, y_train


# %%
import seaborn as sns
sns.set_context("talk")

X_train, X_test, y_train = generate_data(n_samples=50)
_ = sns.scatterplot(
    x=X_train["Feature"], y=y_train, color="black", alpha=0.5)

# %% [markdown]
# The link between our feature and the target to predict is non-linear.
# However, a decision tree is capable of fitting such data.

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# %%
ax = sns.scatterplot(
    x=X_train["Feature"], y=y_train, color="black", alpha=0.5)
ax.plot(X_test, y_pred, label="Fitted tree")
_ = ax.legend()

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
# We will create a function that given `X` and `y` will return a bootstrap
# sample `X_bootstrap` and `y_bootstrap`.


# %%
def bootstrap_sample(X, y):
    bootstrap_indices = rng.choice(
        np.arange(y.shape[0]), size=y.shape[0], replace=True,
    )
    X_bootstrap_sample = X.iloc[bootstrap_indices]
    y_bootstrap_sample = y.iloc[bootstrap_indices]
    return X_bootstrap_sample, y_bootstrap_sample


# %% [markdown]
# We will generate 3 bootstrap samples and qualitatively check the difference
# with the original dataset.

# %%
import matplotlib.pyplot as plt

n_bootstrap = 3
_, ax = plt.subplots(figsize=(8, 6))

for marker, bootstrap_idx in zip(["o", "^", "x"], range(n_bootstrap)):
    X_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(
        X_train, y_train)
    sns.scatterplot(
        x=X_bootstrap_sample["Feature"], y=y_bootstrap_sample,
        label=f"Bootstrap sample #{bootstrap_idx}", marker=marker,
        alpha=0.5, ax=ax)


# %% [markdown]
# We observe that the 3 generated bootstrap samples are all different. To
# confirm this intuition, we can check the number of unique samples in the
# bootstrap samples.

# %%
# we need to generate a larger set to have a good estimate
X_huge_train, y_huge_train, X_test_huge = generate_data(n_samples=10000)
X_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(
    X_huge_train, y_huge_train)

print(
    f"Percentage of samples present in the original dataset: "
    f"{np.unique(X_bootstrap_sample).size / X_bootstrap_sample.size * 100:.1f}"
    f"%"
)

# %% [markdown]
# On average, 63.2% of the original data points of the original dataset will
# be present in the bootstrap sample. The other 36.8% are just repeated
# samples.
#
# So, we are able to generate many datasets, all slightly different. Now, we
# can fit a decision tree to each of these datasets and each decision
# tree shall be slightly different as well.

# %%
_, axs = plt.subplots(
    ncols=n_bootstrap, figsize=(16, 6), sharex=True, sharey=True,
)

forest = []
for idx, (ax, _) in enumerate(zip(axs, range(n_bootstrap))):
    X_bootstrap_sample, y_bootstrap_sample = bootstrap_sample(
        X_train, y_train)
    forest.append(
        DecisionTreeRegressor(max_depth=3, random_state=0).fit(
            X_bootstrap_sample, y_bootstrap_sample
        )
    )

    y_pred = forest[-1].predict(X_test)

    sns.scatterplot(
        x=X_bootstrap_sample["Feature"], y=y_bootstrap_sample, ax=ax,
        color="black", alpha=0.5)
    ax.plot(X_test, y_pred, linewidth=3, label="Fitted tree")
    ax.set_title(f"Bootstrap sample #{idx}")
    ax.legend()

# %% [markdown]
# We can plot these decision functions on the same plot to see the difference.

# %%
ax = sns.scatterplot(
    x=X_train["Feature"], y=y_train, color="black", alpha=0.5)
y_pred_forest = []
for tree_idx, tree in enumerate(forest):
    y_pred = tree.predict(X_test)
    ax.plot(X_test, y_pred, label=f"Tree #{tree_idx} predictions",
            linestyle="--", linewidth=3, alpha=0.8)
    y_pred_forest.append(y_pred)
_ = plt.legend()

# %% [markdown]
# ## Aggregating
#
# Once our trees are fitted and we are able to get predictions for each of
# them, we also need to combine them. In regression, the most straightforward
# approach is to average the different predictions from all learners. We can
# plot the averaged predictions from the previous example.

# %%
ax = sns.scatterplot(
    x=X_train["Feature"], y=y_train, color="black", alpha=0.5)
y_pred_forest = []
for tree_idx, tree in enumerate(forest):
    y_pred = tree.predict(X_test)
    ax.plot(X_test, y_pred, label=f"Tree #{tree_idx} predictions",
            linestyle="--", linewidth=3, alpha=0.8)
    y_pred_forest.append(y_pred)

y_pred_forest = np.mean(y_pred_forest, axis=0)
ax.plot(X_test, y_pred_forest, label="Averaged predictions",
        linestyle="-", linewidth=3, alpha=0.8)
_ = plt.legend()

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

bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(), n_estimators=3)
bagging.fit(X_train, y_train)
y_pred_forest = bagging.predict(X_test)

# %%
ax = sns.scatterplot(
    x=X_train["Feature"], y=y_train, color="black", alpha=0.5)
ax.plot(X_test, y_pred_forest, label="Bag of decision trees",
        linestyle="-", linewidth=3, alpha=0.8)
_ = plt.legend()

# %% [markdown]
# While we used a decision tree as a base model, nothing prevent us of using
# any other type of model. We will give an example that will use a linear
# regression.

# %%
from sklearn.linear_model import LinearRegression

bagging = BaggingRegressor(
    base_estimator=LinearRegression(), n_estimators=3)
bagging.fit(X_train, y_train)
y_pred_linear = bagging.predict(X_test)

# %%
ax = sns.scatterplot(
    x=X_train["Feature"], y=y_train, color="black", alpha=0.5)
ax.plot(X_test, y_pred_forest, label="Bag of decision trees",
        linestyle="-", linewidth=3, alpha=0.8)
ax.plot(X_test, y_pred_linear, label="Bag of linear regression",
        linestyle="-", linewidth=3, alpha=0.8)
_ = plt.legend()

# %% [markdown]
# However, we see that using a bag of linear models is not helpful here because
# we still obtain a linear model.
