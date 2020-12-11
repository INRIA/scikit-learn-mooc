# %% [markdown]
# # 📃 Solution for introductory example for hyperparameters tuning
#
# In this exercise, we aim at showing the effect on changing hyperparameter
# value of predictive pipeline. As an illustration, we will use a linear model
# only on the numerical features of adult census to simplify the pipeline.
#
# Let's start by loading the data.

# %%
from sklearn import set_config
set_config(display='diagram')

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

target = df[target_name]
data = df[numerical_columns]

# %% [markdown]
# We will first divide the data into a train and test set to evaluate
# the model.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# First, define a logistic regression with a preprocessing stage to scale the
# data.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression()),
])
model

# %% [markdown]
# Now, fit the model on the train set and compute the model's accuracy on the
# test set.

# %%
model.fit(data_train, target_train)
accuracy = model.score(data_test, target_test)
print(f"Accuracy of the model is: {accuracy:.3f}")

# %% [markdown]
# We will use this model as a baseline. Now, we will check the effect of
# changing the value of the hyperparameter `C` in logistic regression. First,
# check what is the default value of the hyperparameter `C` of the logistic
# regression.

# %%
print(f"The hyperparameter C was: {model[-1].C}")

# %% [markdown]
# Create a model by setting the `C` hyperparameter to `0.001` and compute the
# performance of the model.

# %%
model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression(C=0.001)),
])
model

# %%
model.fit(data_train, target_train)
accuracy = model.score(data_test, target_test)
print(f"Accuracy of the model is: {accuracy:.3f}")

# %% [markdown]
# We observe that the performance of the model decreased. Repeat the same
# experiment for `C=100`

# %%
model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression(C=100)),
])
model

# %%
model.fit(data_train, target_train)
accuracy = model.score(data_test, target_test)
print(f"Accuracy of the model is: {accuracy:.3f}")

# %% [markdown]
# We see that the performance of the model in this case is as good as the
# original model. However, we don't know if there is a value for `C` in the
# between 0.001 and 100 that will lead to a better model.
#
# You can try by hand a couple of values in this range to see if you can
# improve the performance.
