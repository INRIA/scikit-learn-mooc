# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using numerical and categorical variables together
#
# In the previous notebooks, we showed the required preprocessing to apply
# when dealing with numerical and categorical variables. However, we decoupled
# the process to treat each type individually. In this notebook, we will show
# how to combine these preprocessing steps.
#
# We will first load the entire adult census dataset.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name, "fnlwgt"])

# %% [markdown]
# ```{caution}
# Here and later, we use the name `data` and `target` to be explicit. In
# scikit-learn, documentation `data` is commonly named `X` and `target` is
# commonly called `y`.
# ```

# %% [markdown]
# We recall that both `"education-num"` and `"education"` contain the same
# information. In the previous notebook, we dropped `"education-num"` and
# used `"education"` instead; we will do the same processing here.

# %%
data = data.drop(columns="education-num")

# %% [markdown]
# ## Selection based on data types
#
# We will separate categorical and numerical variables using their data
# types to identify them, as we saw previously that `object` corresponds
# to categorical columns (strings). We make use of `make_column_selector`
# helper to select the corresponding columns.

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

# %% [markdown]
# ## Dispatch columns to a specific processor
#
# In the previous sections, we saw that we need to treat data differently
# depending on their nature (i.e. numerical or categorical).
#
# Scikit-learn provides a `ColumnTransformer` class which will send specific
# columns to a specific transformer, making it easy to fit a single predictive
# model on a dataset that combines both kinds of variables together
# (heterogeneously typed tabular data).
#
# We first define the columns depending on their data type:
#
# * **one-hot encoding** will be applied to categorical columns. Besides, we
#   use `handle_unknown="ignore"` to solve the potential issues due to rare
#   categories.
# * **numerical scaling** numerical features which will be standardized.
#
# Now, we create our `ColumnTransfomer` by specifying three values:
# the preprocessor name, the transformer, and the columns.
# First, let's create the preprocessors for the numerical and categorical
# parts.

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

# %% [markdown]
# Now, we create the transformer and associate each of these preprocessors
# with their respective columns.

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard-scaler', numerical_preprocessor, numerical_columns)])

# %% [markdown]
# We can take a minute to represent graphically the structure of a
# `ColumnTransformer`:
#
# ![columntransformer diagram](../figures/api_diagram-columntransformer.svg)
#
# The `ColumnTransformer` will be a transformer that should do the following
# steps:
#
# * **split the columns** of the original dataset based on the column names
#   or indices provided. Thus, we will obtain as many subsets as the number of
#   transformers in the `ColumnTransformer`;
# * **transform each subset of data**. Indeed, each subset that has been
#   created will is affected a specific transformer. Each transformer will
#   internally call `fit_transform` or `transform`. Thus, the output of this
#   transformation will be a set of transformed dataset.
# * **concatenate all transformed dataset**. Once all input data transformed by
#   their respective transformers, all datasets are concatenated and provided
#   as an output.
#
# Thus, a `ColumnTransformer` is just another transformer: it get some input
# data and will output some transformed data. However, columns will be
# transformed differently depending on how a user defines the relationship
# between columns and transformers.
#
# Now, that we know that our `preprocessor` can be seen as any scikit-learn
# transformer, we define a pipeline to stack our `preprocessor` with our
# classifier (logistic regression).

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

# %% [markdown]
# Starting from `scikit-learn 0.23`, the notebooks can display an interactive
# view of the pipelines.

# %%
from sklearn import set_config
set_config(display='diagram')
model

# %% [markdown]
# The final model is more complex than the previous models but still follows
# the same API (the same set of methods that can be called by the user):
#
# - the `fit` method is called to preprocess the data then train the
#   classifier;
# - the `predict` method make predictions on new data;
# - the `score` method is used to predict on the test data and compare the
#   predictions to the expected test labels to compute the accuracy.
#
# Let's start by splitting our data into train and test sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
#
# ```{caution}
# Be aware that we use `train_test_split` here for didactic purposes, to show
# the scikit-learn API.
# ```
#
# Now, we can train the model on the train set.

# %%
_ = model.fit(data_train, target_train)

# %% [markdown]
# Then, we can send the raw dataset straight to the pipeline. Indeed, we do not
# need to make any manual preprocessing (calling the `transform` or
# `fit_transform` methods) as it will be handled when calling the `predict`
# method. As an example, we predict on the five first samples from the test
# set.

# %%
data_test.head()

# %%
model.predict(data_test)[:5]

# %%
target_test[:5]

# %% [markdown]
# To get directly the accuracy score, we need to call the `score` method. Let's
# compute the accuracy score on the entire test set.

# %%
model.score(data_test, target_test)

# %% [markdown]
# ## Evaluation of the model with cross-validation
#
# As previously stated, a predictive model should be evaluated by
# cross-validation. Our model is usable with the cross-validation tools of
# scikit-learn as any other predictors:

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
cv_results

# %%
scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# The compound model has a higher predictive accuracy than the two models that
# used numerical and categorical variables in isolation.

# %% [markdown]
# ## Fitting a more powerful model
#
# **Linear models** are nice because they are usually cheap to train,
# **small** to deploy, **fast** to predict and give a **good baseline**.
#
# However, it is often useful to check whether more complex models such as an
# ensemble of decision trees can lead to higher predictive performance.
#
# In the following cell we try a scalable implementation of the **Gradient
# Boosting Machine** algorithm. For this class of models, we know that contrary
# to linear models, it is **useless to scale the numerical features** and
# furthermore it is both safe and significantly more computationally efficient
# to use an arbitrary **integer encoding for the categorical variables** even
# if the ordering is arbitrary. Therefore we adapt the preprocessing pipeline
# as follows:

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)

preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns)],
    remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

# %% [markdown]
# Now that we created our model, we can check its statistical performance.

# %%
# %%time
_ = model.fit(data_train, target_train)

# %%
model.score(data_test, target_test)

# %% [markdown]
# We can observe that we get significantly higher accuracies with the Gradient
# Boosting model. This is often what we observe whenever the dataset has a
# large number of samples and limited number of informative features (e.g. less
# than 1000) with a mix of numerical and categorical variables.
#
# This explains why Gradient Boosted Machines are very popular among
# datascience practitioners who work with tabular data.

# %% [markdown]
# In this notebook we:
#
# * used a `ColumnTransformer` to apply different preprocessing for
#   categorical and numerical variables;
# * used a pipeline to chain the `ColumnTransformer` preprocessing and
#   logistic regresssion fitting;
# * seen that **gradient boosting methods** can outperforms the basic linear
#   approach.
