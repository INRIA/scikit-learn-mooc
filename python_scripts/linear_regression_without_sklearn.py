# %% [markdown]
# # Linear regression without scikit-learn
#
# In this notebook, we introduce linear regression. Before to present class
# available in scikit-learn, we present some insights with a simple example.
# We will use a dataset that contains information about penguins.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")
data.head()

# %% [markdown]
# This dataset contains measurements taken of penguins. We will formulate the
# following problem: using the flipper length of a penguin, we would like
# to infer its mass.

# %%
import seaborn as sns
sns.set_context("talk")

feature_names = "Flipper Length (mm)"
target_name = "Body Mass (g)"
X, y = data[[feature_names]], data[target_name]

_ = sns.scatterplot(data=data, x=feature_names, y=target_name)

# %% [markdown]
# In this problem, penguin mass is our target. It is a continuous
# variable that roughly varies between 2700 g and 6300 g. Thus, this is a
# regression problem (in contrast to classification). We also see that there is
# almost a linear relationship between the body mass of the penguin and the
# flipper length. The longer the flipper, the heavier the penguin.
#
# Thus, we could come up with a simple formula, where given a flipper length
# we could compute the body mass of a penguin using a linear relationship of
# of the form `y = a * x + b` where `a` and `b` are the 2 parameters of our
# model.


# %%
def linear_model_flipper_mass(
    flipper_length, weight_flipper_length, intercept_body_mass
):
    """Linear model of the form y = a * x + b"""
    body_mass = weight_flipper_length * flipper_length + intercept_body_mass
    return body_mass


# %% [markdown]
# Using the model we defined above, we can check the body mass values
# predicted for a range of flipper lengths. We will set `weight_flipper_length`
# to be 45 and `intercept_body_mass` to be -5000.

# %%
import numpy as np

weight_flipper_length = 45
intercept_body_mass = -5000

flipper_length_range = np.linspace(X.min(), X.max(), num=300)
predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)

label = ("{0:.2f} (g / mm) * flipper length + "
         "{1:.2f} (g)")
ax = sns.scatterplot(data=data, x=feature_names, y=target_name)
ax.plot(
    flipper_length_range, predicted_body_mass,
    label=label.format(weight_flipper_length, intercept_body_mass),
    linewidth=4)
_ = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.2), ncol=1)

# %% [markdown]
# The variable `weight_flipper_length` is a weight applied to the feature
# `flipper_length` in
# order to make the inference. When this coefficient is positive, it means that
# penguins with longer flipper lengths will have larger body masses.
# If the coefficient is negative, it means that penguins with shorter flipper
# flipper lengths have larger body masses. Graphically, this coefficient is
# represented by the slope of the curve in the plot. Below we show what the
# curve would look like when the `weight_flipper_length` coefficient is
# negative.

# %%
weight_flipper_length = -40
intercept_body_mass = 13000

predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)

label = ("{0:.2f} (g / mm) * flipper length + "
         "{1:.2f} (g)")
ax = sns.scatterplot(data=data, x=feature_names, y=target_name)
ax.plot(
    flipper_length_range, predicted_body_mass,
    label=label.format(weight_flipper_length, intercept_body_mass),
    linewidth=4)
_ = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.2), ncol=1)


# %% [markdown]
# In our case, this coefficient has a meaningful unit: g/mm.
# For instance, a coefficient of 40 g/mm, means that for each
# additional millimeter in flipper length, the body weight predicted will
# increase by 40 g.

# %%
body_mass_180 = linear_model_flipper_mass(
    flipper_length=180, weight_flipper_length=40, intercept_body_mass=0
)
body_mass_181 = linear_model_flipper_mass(
    flipper_length=181, weight_flipper_length=40, intercept_body_mass=0
)

print(
    f"The body mass for a flipper length of 180 mm is {body_mass_180} g and "
    f"{body_mass_181} g for a flipper length of 181 mm"
)

# %% [markdown]
# We can also see that we have a parameter `intercept_body_mass` in our model.
# This parameter corresponds to the value on the y-axis if `flipper_length=0`
# (which in our case is only a mathematical consideration, as in our data,
#  the value of `flipper_length` only goes from 170mm to 230mm). This y-value
# when x=0 is called the y-intercept. If `intercept_body_mass` is 0, the curve
# will pass through the origin:

# %%
weight_flipper_length = 25
intercept_body_mass = 0

predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)

label = ("{0:.2f} (g / mm) * flipper length + "
         "{1:.2f} (g)")
ax = sns.scatterplot(data=data, x=feature_names, y=target_name)
ax.plot(
    flipper_length_range, predicted_body_mass,
    label=label.format(weight_flipper_length, intercept_body_mass),
    linewidth=4)
_ = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.2), ncol=1)

# %% [markdown]
# Otherwise, it will pass through the `intercept_body_mass` value:

# %%
weight_flipper_length = 45
intercept_body_mass = -5000

predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)

label = ("{0:.2f} (g / mm) * flipper length + {1:.2f} (g)")
ax = sns.scatterplot(data=data, x=feature_names, y=target_name)
ax.plot(
    flipper_length_range, predicted_body_mass,
    label=label.format(weight_flipper_length, intercept_body_mass),
    linewidth=4)
_ = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.2), ncol=1)

# %% [markdown]
#  In this notebook, we have seen the parametrization of a linear regression
#  model and more precisely meaning of the terms weights and intercepts.
