# %% [markdown]
# # Exercise 01
#
# The aim of this exercise is three-fold:
#
# * understand the parametrization of a linear model;
# * quantify the goodness of fit of a set of such model.
#
# We will reuse part of the code of the course to:
#
# * load data;
# * create the function representing a linear model;
# * plot the data and the linear model function.
#
# ## Prerequisites
#
# ### Data loading

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")
feature_names = "Flipper Length (mm)"
target_name = "Body Mass (g)"
X, y = data[[feature_names]], data[target_name]

# %% [markdown]
# ### Model definition


# %%
def linear_model_flipper_mass(
    flipper_length, weight_flipper_length, intercept_body_mass
):
    """Linear model of the form y = a * x + b"""
    body_mass = weight_flipper_length * flipper_length + intercept_body_mass
    return body_mass


# %%
import seaborn as sns
import matplotlib.pyplot as plt


# %% [markdown]
# ### Data and model visualization


def plot_data_and_model(
    flipper_length_range, weight_flipper_length, intercept_body_mass,
    ax=None,
):
    """Compute and plot the prediction."""
    inferred_body_mass = linear_model_flipper_mass(
        flipper_length_range,
        weight_flipper_length=weight_flipper_length,
        intercept_body_mass=intercept_body_mass,
    )

    if ax is None:
        _, ax = plt.subplots()

    sns.scatterplot(data=data, x=feature_names, y=target_name, ax=ax)
    ax.plot(
        flipper_length_range,
        inferred_body_mass,
        linewidth=3,
        label=(
            f"{weight_flipper_length:.2f} (g / mm) * flipper length + "
            f"{intercept_body_mass:.2f} (g)"
        ),
    )
    plt.legend()


# %% [markdown]
# ## Main exercise
#
# ### Question 1.
#
# Given a vector of the flipper length, several weights and intercepts to
# plot several linear model that could fit our data. Use the above
# visualization helper function to visualize both the model and data.

# %%
import numpy as np

flipper_length_range = np.linspace(X.min(), X.max(), num=300)

# %%
# TODO
# weights = [...]
# intercepts = [...]


# %% [markdown]
# ### Question 2.
#
# In the previous question, you were asked to create several linear models.
# The visualization allowed you to qualitatively assess if a model was better
# than another.
#
# Now, you should come up with a quantitative measure which will indicate the
# goodness of fit of each linear model. This quantitative metric should result
# in a single scalar and allow you to pick up the best model.


# %%
def goodness_fit_measure(true_values, predictions):
    # TODO: define a measure indicating the goodness of fit of a model given
    # the true values and the model predictions.
    pass


# %%
# TODO: uncomment
# for model_idx, (weight, intercept) in enumerate(zip(weights, intercepts)):
#     y_pred = linear_model_flipper_mass(X, weight, intercept)
#     print(f"Model #{model_idx}:")
#     print(f"{weight:.2f} (g / mm) * flipper length + {intercept:.2f} (g)")
#     print(f"Error: {goodness_fit_measure(y, y_pred):.3f}\n")

# %%
