# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# TODO add intro with objectives

# ## [markdown]
# Let's first load the data as we did in the previous notebook. TODO add link.

# %%
import pandas as pd

df = pd.read_csv("https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

target_name = "class"
target = df[target_name].to_numpy()

data = df.drop(columns=[target_name, "fnlwgt"])

# %% [markdown]
# ## Working with categorical variables
#
# As we have seen in the previous section, a numerical variable is a continuous
# quantity represented by a real or integer number. Those variables can be
# naturally handled by machine learning algorithms that typically composed of
# a sequence of arithmetic instructions such as additions and multiplications.
#
# By opposition, categorical variables have discrete values typically represented
# by string labels taken in a finite list of possible choices. For instance, the
# variable `native-country` in our dataset is a categorical variable because it
# encodes the data using a finite list of possible countries (along with the `?`
# marker when this information is missing):

# %%
data["native-country"].value_counts()

# %% [markdown]
# In the remainder of this section, we will present different strategies to
# encode categorical data into numerical data which can be used by a
# machine-learning algorithm.

# %%
data.dtypes


# %%
categorical_columns = [c for c in data.columns
                       if data[c].dtype.kind not in ["i", "f"]]
categorical_columns

# %%
data_categorical = data[categorical_columns]
data_categorical.head()


# %%
print(f"The datasets is composed of {data_categorical.shape[1]} features")


# %% [markdown]
# ### Encoding ordinal categories
#
# The most intuitive strategy is to encode each category with a number.
# The `OrdinalEncoder` will transform the data in such manner.


# %%
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
data_encoded[:5]


# %% [markdown]
# We can see that all categories have been encoded for each feature
# independently. We can also notice that the number of features before and after
# the encoding is the same.
#
# However, one has to be careful when using this encoding strategy. Using this
# integer representation can lead the downstream models to make the assumption
# that the categories are ordered: 0 is smaller than 1 which is smaller than 2,
# etc.
#
# By default, `OrdinalEncoder` uses a lexicographical strategy to map string
# category labels to integers. This strategy is completely arbitrary and often be
# meaningless. For instance suppose the dataset has a categorical variable named
# "size" with categories such as "S", "M", "L", "XL". We would like the integer
# representation to respect the meaning of the sizes by mapping them to increasing
# integers such as 0, 1, 2, 3. However lexicographical strategy used by default
# would map the labels "S", "M", "L", "XL" to 2, 1, 0, 3.
#
# The `OrdinalEncoder` class accepts a "categories" constructor argument to pass
# an the correct ordering explicitly.
#
# If a categorical variable does not carry any meaningful order information then
# this encoding might be misleading to downstream statistical models and you might
# consider using one-hot encoding instead (see below).
#
# Note however that the impact a violation of this ordering assumption is really
# dependent on the downstream models (for instance linear models are much more
# sensitive than models built from a ensemble of decision trees).
#
# ### Encoding nominal categories (without assuming any order)
#
# `OneHotEncoder` is an alternative encoder that can prevent the dowstream
# models to make a false assumption about the ordering of categories. For a
# given feature, it will create as many new columns as there are possible
# categories. For a given sample, the value of the column corresponding to the
# category will be set to `1` while all the columns of the other categories will
# be set to `0`.

# %%
print(f"The dataset is composed of {data_categorical.shape[1]} features")
data_categorical.head()


# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_categorical)
print(f"The dataset encoded contains {data_encoded.shape[1]} features")
data_encoded


# %% [markdown]
# Let's wrap this numpy array in a dataframe with informative column names as provided by the encoder object:

# %%
columns_encoded = encoder.get_feature_names(data_categorical.columns)
pd.DataFrame(data_encoded, columns=columns_encoded).head()


# %% [markdown]
# Look at how the workclass variable of the first 3 records has been encoded and compare this to the original string representation.
#
# The number of features after the encoding is than 10 times larger than in the
# original data because some variables such as `occupation` and `native-country`
# have many possible categories.
#
# We can now integrate this encoder inside a machine learning pipeline as in the
# case with numerical data: let's train a linear classifier on
# the encoded data and check the performance of this machine learning pipeline
# using cross-validation.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = make_pipeline(
    OneHotEncoder(handle_unknown='ignore'),
    LogisticRegression(solver='lbfgs', max_iter=1000)
)
scores = cross_val_score(model, data_categorical, target)
print(f"The different scores obtained are: \n{scores}")


# %%
print(f"The accuracy is: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# As you can see, this representation of the categorical variables of the data is slightly more predictive of the revenue than the numerical variables that we used previously.

# %% [markdown]
# ## Exercise 1:
#
# - Try to fit a logistic regression model on categorical data transformed by
#   the OrdinalEncoder instead. What do you observe?
#
# Use the dedicated notebook to do this exercise.


# %% [markdown]
# ## Using numerical and categorical variables together
#
# In the previous sections, we saw that we need to treat data specifically
# depending of their nature (i.e. numerical or categorical).
#
# Scikit-learn provides a `ColumnTransformer` class which will dispatch some
# specific columns to a specific transformer making it easy to fit a single
# predictive model on a dataset that combines both kinds of variables together
# (heterogeneously typed tabular data).
#
# We can first define the columns depending on their data type:
# * **binary encoding** will be applied to categorical columns with only too
#   possible values (e.g. sex=male or sex=female in this example). Each binary
#   categorical columns will be mapped to one numerical columns with 0 or 1
#   values.
# * **one-hot encoding** will be applied to categorical columns with more that
#   two possible categories. This encoding will create one additional column for
#   each possible categorical value.
# * **numerical scaling** numerical features which will be standardized.
#
#
#
#
#
#
#


# %%
binary_encoding_columns = ['sex']
one_hot_encoding_columns = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship',
                            'race', 'native-country']
scaling_columns = ['age', 'education-num', 'hours-per-week',
                   'capital-gain', 'capital-loss']

# %% [markdown]
# We can now create our `ColumnTransfomer` by specifying a list of triplet
# (preprocessor name, transformer, columns). Finally, we can define a pipeline
# to stack this "preprocessor" with our classifier (logistic regression).

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])
model = make_pipeline(
    preprocessor,
    LogisticRegression(solver='lbfgs', max_iter=1000)
)

# %% [markdown]
# The final model is more complex than the previous models but still follows the
# same API:
# - the `fit` method is called to preprocess the data then train the classifier;
# - the `predict` method can make predictions on new data;
# - the `score` method is used to predict on the test data and compare the
#   predictions to the expected test labels to compute the accuracy.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)
model.fit(data_train, target_train)
model.predict(data_test)[:5]


# %%
target_test[:5]


# %%
data_test.head()


# %%
model.score(data_test, target_test)


# %% [markdown]
# This model can also be cross-validated as usual (instead of using a single
# train-test split):

# %%
scores = cross_val_score(model, data, target, cv=5)
print(f"The different scores obtained are: \n{scores}")


# %%
print(f"The accuracy is: {scores.mean():.3f} +- {scores.std():.3f}")

# %% [markdown]
# The compound model has a higher predictive accuracy than the
# two models that used numerical and categorical variables in
# isolation.

# %% [markdown]
# # Fitting a more powerful model
#
# Linear models are very nice because they are usually very cheap to train,
# small to deploy, fast to predict and give a good baseline.
#
# However it is often useful to check whether more complex models such as
# ensemble of decision trees can lead to higher predictive performance.
#
# In the following we try a scalable implementation of the Gradient Boosting
# Machine algorithm. For this class of models, we know that contrary to linear
# models, it is useless to scale the numerical features and furthermore it is
# both safe and significantly more computationally efficient use an arbitrary
# integer encoding for the categorical variable even if the ordering is
# arbitrary. Therefore we adapt the preprocessing pipeline as follows:

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# For each categorical column, extract the list of all possible categories
# in some arbritrary order.
categories = [data[column].unique() for column in data[categorical_columns]]

preprocessor = ColumnTransformer([
    ('categorical', OrdinalEncoder(categories=categories), categorical_columns),
], remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())
model.fit(data_train, target_train)
print(model.score(data_test, target_test))


# %% [markdown]
# We can observe that we get significantly higher accuracies with the Gradient
# Boosting model. This is often what we observe whenever the dataset has a large
# number of samples and limited number of informative features (e.g. less than
# 1000) with a mix of numerical and categorical variables.
#
# This explains why Gradient Boosted Machines are very popular among datascience
# practitioners who work with tabular data.
#
#
#
#
#
#
#


# %% [markdown]
# ## Exercise 2:
#
# - Check that scaling the numerical features does not impact the speed or
#   accuracy of HistGradientBoostingClassifier
# - Check that one-hot encoding the categorical variable does not improve the
#   accuracy of HistGradientBoostingClassifier but slows down the training.
#
# Use the dedicated notebook to do this exercise.
