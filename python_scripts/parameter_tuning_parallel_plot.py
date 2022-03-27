# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Analysis of hyperparameter search results

# %% [markdown]
# In the previous notebook we showed how to implement a randomized
# search for tuning the hyperparameters of a `HistGradientBoostingClassifier`
# to fit the `adult_census` dataset. In practice, a randomized hyperparameter
# search is usually run with a large number of iterations.

# %% [markdown]
# In order to avoid the computational cost and still make a decent analysis,
# we load the results obtained from a similar search with 500 iterations.

# %%
import pandas as pd

cv_results = pd.read_csv("../figures/randomized_search_results.csv", index_col=0)
cv_results

# %% [markdown]
# We define a function to remove the prefixes in the hyperparameters
# column names.

# %%
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

cv_results = cv_results.rename(shorten_param, axis=1)
cv_results

# %% [markdown]
# As we have more than 2 parameters in our randomized-search, we
# cannot visualize the results using a heatmap. We could still do
# it pair-wise, but having a two-dimensional projection of a
# multi-dimensional problem can lead to a wrong interpretation of
# the scores.

# %%
import seaborn as sns
import numpy as np

df = pd.DataFrame(
    {
        "max_leaf_nodes": cv_results["max_leaf_nodes"],
        "learning_rate": cv_results["learning_rate"],
        "score_bin": pd.cut(
            cv_results["mean_test_score"], bins=np.linspace(0.5, 1.0, 6)
        ),
    }
)
sns.set_palette("YlGnBu_r")
ax = sns.scatterplot(
    data=df,
    x="max_leaf_nodes",
    y="learning_rate",
    hue="score_bin",
    s=50,
    color="k",
    edgecolor=None,
)
ax.set_xscale("log")
ax.set_yscale("log")

_ = ax.legend(title="mean_test_score", loc="center left", bbox_to_anchor=(1, 0.5))

# %% [markdown]
# In the previous plot we see that the top performing values are located in a
# band of learning rate between 0.01 and 1.0, but we have no control in how the
# other hyperparameters interact with such values for the learning rate.
# Instead, we can visualize all the hyperparameters at the same time using a
# parallel coordinates plot.

# %%
import numpy as np
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.rename(shorten_param, axis=1).apply(
        {
            "learning_rate": np.log10,
            "max_leaf_nodes": np.log2,
            "max_bins": np.log2,
            "min_samples_leaf": np.log10,
            "l2_regularization": np.log10,
            "mean_test_score": lambda x: x,
        }
    ),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()

# %% [markdown]
# ```{note}
# We **transformed most axis values by taking a log10 or log2** to
# spread the active ranges and improve the readability of the plot.
# ```
#
# The parallel coordinates plot will display the values of the hyperparameters
# on different columns while the performance metric is color coded. Thus, we are
# able to quickly inspect if there is a range of hyperparameters which is
# working or not.
#
# It is possible to **select a range of results by clicking and holding on any
# axis** of the parallel coordinate plot. You can then slide (move) the range
# selection and cross two selections to see the intersections. You can undo a
# selection by clicking once again on the same axis.
#
# In particular for this hyperparameter search, it is interesting to confirm
# that the yellow lines (top performing models) all reach intermediate values
# for the learning rate, that is, tick values between -2 and 0 which correspond
# to learning rate values of 0.01 to 1.0 once we invert back the log10 transform
# for that axis.
#
# But now we can also observe that it is not possible to select the highest
# performing models by selecting lines of on the `max_bins` axis with tick
# values between 1 and 3.
#
# The other hyperparameters are not very sensitive. We can check that if we
# select the `learning_rate` axis tick values between -1.5 and -0.5 and
# `max_bins` tick values between 5 and 8, we always select top performing
# models, whatever the values of the other hyperparameters.

# %% [markdown]
#
# In this notebook, we saw how to interactively explore the results of a
# large randomized search with multiple interacting hyperparameters.
# In particular we observed that some hyperparameters have very little
# impact on the cross-validation score, while others have to be adjusted
# within a specific range to get models with good predictive accuracy.
