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
# # Encoding of categorical variables
#
# In this notebook, we will present typical ways to deal with **categorical
# variables**, namely **ordinal encoding** and **one-hot encoding**.

# %% [markdown]
# Let's first load the entire adult dataset containing both numerical and
# categorical data.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = df[target_name]

data = df.drop(columns=[target_name, "fnlwgt"])

# %% [markdown]
# ## Working with categorical variables
#
# As we have seen in the previous section, a numerical variable is a continuous
# quantity represented by a real or integer number. These variables can be
# naturally handled by machine learning algorithms that are typically composed
# of a sequence of arithmetic instructions such as additions and
# multiplications.
#
# In contrast, categorical variables have discrete values, typically
# represented by string labels taken from a finite list of possible choices.
# For instance, the variable `native-country` in our dataset is a categorical
# variable because it encodes the data using a finite list of possible
# countries (along with the `?` symbol when this information is missing):

# %%
data["native-country"].value_counts()

# %% [markdown]
# One might question how could we easily recognize categorical columns. One
# useful tip is to look at the data type of each column.

# %%
data.dtypes

# %% [markdown]
# If we look at the "native-country" column data type, we observe it is of
# `object` data type. The reason is that categories are represented with
# string.
#
# Sometimes, categorical columns could also be encoded with integer. In this
# case, looking at the data type will not be enough. In the exploration and in
# the previous notebook, we saw that the column `"education-num"` is such a
# column.

# %%
data["education-num"].value_counts()

# %% [markdown]
# When considering categorical columns, we should include this columns.
# However, we also saw that `"education-num"` and `"education"` columns
# represent exactly the same information. Therefore, we can use one of the two
# columns. So, we can make the choice to consider only `object` dtype columns.
#
# In the previous notebook, we manually defined the numerical columns. We could
# to the same here. However, since that we can select columns based on a
# specific dtypes, we can use a scikit-learn helper allowing to select the
# column based on their data types. It is called `make_column_selector`. We
# will illustrate how to use this helper.

# %%
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns

# %% [markdown]
# We can create the selector by passing the data type to include or exclude.
# Once the selector created, we can pass the input dataset and the name of
# the selected columns will be returned.

# %%
data_categorical = data[categorical_columns]
data_categorical.head()

# %%
print(f"The dataset is composed of {data_categorical.shape[1]} features")

# %% [markdown]
# In the remainder of this section, we will present different strategies to
# encode categorical data into numerical data which can be used by a
# machine-learning algorithm.

# %% [markdown]
# ### Encoding ordinal categories
#
# The most intuitive strategy is to encode each category with a different
# number. The `OrdinalEncoder` will transform the data in such manner.

# %%
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]

# %%
print(
    f"The dataset encoded contains {data_encoded.shape[1]} features")

# %% [markdown]
# We can see that the categories have been encoded for each feature (column)
# independently. We can also note that the number of features before and after
# the encoding is the same.
#
# However, one has to be careful when using this encoding strategy. Using this
# integer representation can lead the downstream models to make the assumption
# that the categories are ordered: 0 is smaller than 1 which is smaller than 2,
# etc.
#
# By default, `OrdinalEncoder` uses a lexicographical strategy to map string
# category labels to integers. This strategy is completely arbitrary and often
# be meaningless. For instance suppose the dataset has a categorical variable
# named "size" with categories such as "S", "M", "L", "XL". We would like the
# integer representation to respect the meaning of the sizes by mapping them to
# increasing integers such as 0, 1, 2, 3. However lexicographical strategy used
# by default would map the labels "S", "M", "L", "XL" to 2, 1, 0, 3.
#
# The `OrdinalEncoder` class accepts a "categories" constructor argument to
# pass in the correct ordering explicitly.
#
# If a categorical variable does not carry any meaningful order information
# then this encoding might be misleading to downstream statistical models and
# you might consider using one-hot encoding instead (see below).
#
# Note however that the impact of violating this ordering assumption is really
# dependent on the downstream models (for instance linear models are much more
# sensitive than models built from a ensemble of decision trees).
#
# ### Encoding nominal categories (without assuming any order)
#
# `OneHotEncoder` is an alternative encoder that can prevent the dowstream
# models to make a false assumption about the ordering of categories. For a
# given feature, it will create as many new columns as there are possible
# categories. For a given sample, the value of the column corresponding to the
# category will be set to `1` while all the columns of the other categories
# will be set to `0`.

# %%
print(
    f"The dataset is composed of {data_categorical.shape[1]} features")
data_categorical.head()

# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]

# %%
print(
    f"The dataset encoded contains {data_encoded.shape[1]} features")

# %% [markdown]
# Let's wrap this numpy array in a dataframe with informative column names as
# provided by the encoder object:

# %%
columns_encoded = encoder.get_feature_names(data_categorical.columns)
pd.DataFrame(data_encoded, columns=columns_encoded).head()

# %% [markdown]
# Look at how the "workclass" variable of the first 3 records has been encoded
# and compare this to the original string representation.
#
# The number of features after the encoding is more than 10 times larger than
# in the original data because some variables such as `occupation` and
# `native-country` have many possible categories.
#
# We can now integrate this encoder inside a machine learning pipeline like we
# did with numerical data: let's train a linear classifier on the encoded data
# and check the performance of this machine learning pipeline using
# cross-validation.
#
# Before to create the pipeline, we have to linger on the `native-country`.
# Let's recall some statistics regarding this column.

# %%
data["native-country"].value_counts()

# %% [markdown]
# We see that the `Holand-Netherlands` category is occuring rarely. This will
# be a problem during cross-validation: if the sample is in the test set then
# the classifier would not have seen the category during training and will not
# be able to encode it.
#
# In scikit-learn, there is two solutions to bypass this issue:
#
# * list all the possible categories and provide it to the `categories`;
# * set the parameter `handle_unknown="ignore"`.
#
# Here, we will use the former strategy because we are going to use it as well
# for the ordinal encoder later on.

# %%
categories = [data_categorical[column].unique()
              for column in data_categorical]

# %% [markdown]
# We can now create our machine learning pipeline.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(categories=categories, drop="if_binary"),
    LogisticRegression())

# %% [markdown]
# Finally, we can check the model's performance only using the categorical
# columns.

# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data_categorical, target)
scores

# %%
print(f"The accuracy is: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# As you can see, this representation of the categorical variables of the data
# is slightly more predictive of the revenue than the numerical variables that
# we used previously.

# %% [markdown]
#
# In this notebook we have:
# * seen two common strategies for encoding categorical features : **ordinal
#   encoding** and **one-hot encoding**;
# * used a pipeline to process **both numerical and categorical** features
#   before fitting a logistic regression.
