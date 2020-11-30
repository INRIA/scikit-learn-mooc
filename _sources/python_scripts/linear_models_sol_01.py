# %% [markdown]
# # 📃 Solution for Exercise 01
#
# The aim of this exercise is three-fold:
#
# * understand the parametrization of a linear model;
# * quantify the goodness of fit of a set of such model.
#
# We will reuse part of the code of the course to:
#
# * load data;
# * create the function representing a linear model.
#
# ## Prerequisite
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


# %% [markdown]
# ## Main exercise
#
# Given a vector of the flipper length, several weights and intercepts to
# plot several linear model that could fit our data. Use the above
# visualization helper function to visualize both the model and data.

# %%
import numpy as np

flipper_length_range = np.linspace(X.min(), X.max(), num=300)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

weights = [-40, 45, 90]
intercepts = [15000, -5000, -14000]

_, ax = plt.subplots()
for weight, intercept in zip(weights, intercepts):
    predicted_body_mass = linear_model_flipper_mass(
        flipper_length_range, weight, intercept)

    label = ("{0:.2f} (g / mm) * flipper length + {1:.2f} (g)")
    sns.scatterplot(data=data, x=feature_names, y=target_name, ax=ax)
    ax.plot(
        flipper_length_range, predicted_body_mass,
        label=label.format(weight, intercept),
        linewidth=4)
    _ = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)

# %% [markdown]
# In the previous question, you were asked to create several linear models.
# The visualization allowed you to qualitatively assess if a model was better
# than another.
#
# Now, you should come up with a quantitative measure which will indicate the
# goodness of fit of each linear model. This quantitative metric should result
# in a single scalar and allow you to pick up the best model.


# %%
def goodness_fit_measure(true_values, predictions):
    # we compute the error between the true values and the predictions of our
    # model
    errors = np.ravel(true_values) - np.ravel(predictions)
    # We have several possible strategy to reduce all errors to a single value.
    # Computing the mean error (sum divided by the number of element) looks
    # like a good solution. However, we have negative error and therefore, we
    # need to sum only positive numbers. Therefore, we can either square each
    # error or take the absolute value: these metrics are known as mean
    # squared error (MSE) and mean absolute error (MAE). Let's use the MAE here
    # as an example.
    return np.mean(np.abs(errors))


# %%
for model_idx, (weight, intercept) in enumerate(zip(weights, intercepts)):
    y_pred = linear_model_flipper_mass(X, weight, intercept)
    print(f"Model #{model_idx}:")
    print(f"{weight:.2f} (g / mm) * flipper length + {intercept:.2f} (g)")
    print(f"Error: {goodness_fit_measure(y, y_pred):.3f}\n")

# %%
