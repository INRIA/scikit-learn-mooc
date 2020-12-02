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
# %%

# %% [markdown]
# # Using numerical and categorical variables together
#
# In this notebook, we will present typical ways to deal with **categorical
# variables**, namely **ordinal encoding** and **one-hot encoding**.
#
# We will load the entire adult census dataset.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = df[target_name]

data = df.drop(columns=[target_name, "fnlwgt"])

# ## [markdown]
# We recall that `"education-num"` and `"education"` columns contains the same
# information. In the previous notebook, we drop `"education-num"` and only
# use the latter column. We will do the same processing here.

# %%
data = data.drop(columns="education-num")

# %% [markdown]
# We separate categorical and numerical variables using the `object` data types
# as we previously saw that it only corresponds to categorical columns. We make
# use of the `make_column_selector` to select the corresponding columns.

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

# %% [markdown]
# Besides, we will list before hand the categories for each categorical column
# to avoid issue with rare categories when evaluating the model during the
# cross-validation.

# %%
categories = [data[column].unique()
              for column in categorical_columns]

# %% [markdown]
# ## ColumnTransformer
#
# In the previous sections, we saw that we need to treat data differently
# depending on their nature (i.e. numerical or categorical).
#
# Scikit-learn provides a `ColumnTransformer` class which will send specific
# columns to a specific transformer, making it easy to fit a single predictive
# model on a dataset that combines both kinds of variables together
# (heterogeneously typed tabular data).
#
# We can first define the columns depending on their data type:
#
# * **one-hot encoding** will be applied to categorical columns. Besides, we
#   will use the option `drop="if_binary"` to drop one of the column since the
#   information will be correlated;
# * **numerical scaling** numerical features which will be standardized.


# %% [markdown]
# We can now create our `ColumnTransfomer` by specifying a list of triplet
# (preprocessor name, transformer, columns). First, let's start by creating
# a transformer for the numerical and categorical part as we did in the
# previous notebooks.

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(categories=categories,
                                         drop="if_binary")
numerical_preprocessor = StandardScaler()

# %% [markdown]
# Now, we can associate each of these preprocessors with their respective
# columns.

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard-scaler', numerical_preprocessor, numerical_columns)])

# %% [markdown]
# Finally, we can define a pipeline to stack this "preprocessor" with our
# classifier (logistic regression).

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
model = make_pipeline(preprocessor, LogisticRegression())

# %% [markdown]
# Starting from `scikit-learn` 0.23, the notebooks can display an interactive
# view of the pipelines.

# %%
from sklearn import config_context
with config_context(display='diagram'):
    model

# %% [markdown]
# The final model is more complex than the previous models but still follows
# the same API:
#
# - the `fit` method is called to preprocess the data then train the
#   classifier;
# - the `predict` method can make predictions on new data;
# - the `score` method is used to predict on the test data and compare the
#   predictions to the expected test labels to compute the accuracy.
#
# Let's start by splitting our data into a train and test sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# Now, we can train the model on the train set.

# %%
_ = model.fit(data_train, target_train)

# %% [markdown]
# Then, we can use the raw dataset directly to the pipeline. Indeed, we don't
# need to make any processing. All the preprocessing will be handle when
# calling `predict`. We will give an example by predicting on the five first
# sample from the test set.

# %%
data_test.head()

# %%
model.predict(data_test)[:5]

# %%
target_test[:5]

# %% [markdown]
# We can get the accuracy score by calling directly the `score` method. We will
# compute the score on the entire test set.

# %%
model.score(data_test, target_test)

# %% [markdown]
# This model can also be cross-validated as we previously did (instead of using
# a single train-test split):

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data, target, cv=5)
scores

# %%
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# The compound model has a higher predictive accuracy than the two models that
# used numerical and categorical variables in isolation.

# %% [markdown]
# ## Fitting a more powerful model
#
# **Linear models** are very nice because they are usually very cheap to train,
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

categorical_preprocessor = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns)],
    remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

# %% [markdown]
# Now that we created our model, we can check the performance of the model.

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
# In this notebook we have:
#
# * used a `ColumnTransformer` to apply different preprocessing for
#   categorical and numerical variables;
# * used a pipeline to chain the `ColumnTransformer` preprocessing and
#   logistic regresssion fitting;
# * seen that **gradient boosting methods** can outperforms the basic linear
#   approach.
