# %%
import numpy as np

rng = np.random.RandomState(0)

X = rng.randn(100, 2)
X[:, 0] += abs(X[:, 0].min()) + 1
X[:, 1] *= 3

# %%
import seaborn as sns

sns.set_context("talk")

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ticks = [-6, -4, -2, 0, 2, 4, 6]

_, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=30, edgecolor="black")
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel("Feature A")
ax.set_ylabel("Feature B")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_xticklabels(ticks)
ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
ax.set_yticklabels(ticks)
ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
ax.grid(visible=True)
ax.set_title("Original dataset\n", loc="center")
plt.savefig("numerical_pipeline_quiz_scaler_original.png", bbox_inches="tight")

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

standard_scaler_mean_only = StandardScaler(with_std=False).fit(X)
standard_scaler_scale_only = StandardScaler(with_mean=False).fit(X)
standard_scaler = StandardScaler().fit(X)
min_max_scaler = MinMaxScaler().fit(X)

# %%
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
for idx, (ax, data) in enumerate(
    zip(
        axs.ravel(),
        [
            standard_scaler_mean_only.transform(X),
            standard_scaler.transform(X),
            min_max_scaler.transform(X),
            standard_scaler_scale_only.transform(X),
        ],
    )
):
    sns.scatterplot(x=data[:, 0], y=data[:, 1], s=30, edgecolor="black", ax=ax)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticklabels(ticks)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.set_yticklabels(ticks)
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.grid(visible=True)
    ax.set_title(f"Preprocessing {'ABCD'[idx]}\n")

fig.subplots_adjust(hspace=0.6, wspace=0.5)
plt.savefig(
    "numerical_pipeline_quiz_scaler_preprocessing.png", bbox_inches="tight"
)

# %%
