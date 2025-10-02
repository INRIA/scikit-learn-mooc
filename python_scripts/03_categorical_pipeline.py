# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Encoding of categorical variables
#
# In this notebook, we present some typical ways of dealing with **categorical
# variables** by encoding them, namely **ordinal encoding** and **one-hot
# encoding**.

# %% [markdown]
# Let's first load the entire adult dataset containing both numerical and
# categorical data.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])

# %% [markdown]
#
# ## Identify categorical variables
#
# As we saw in the previous section, a numerical variable is a quantity
# represented by a continous or integer number. These variables can be naturally
# handled by machine learning algorithms that are typically composed of a
# sequence of arithmetic instructions such as additions and multiplications.
#
# In contrast, categorical variables have discrete values, typically
# represented by string labels (but not only) taken from a finite list of
# possible choices. For instance, the variable `native-country` in our dataset
# is a categorical variable because it encodes the data using a finite list of
# possible countries (along with the `?` symbol when this information is
# missing):

# %%
data["native-country"].value_counts().sort_index()

# %% [markdown]
# How can we easily recognize categorical columns among the dataset? Part of
# the answer lies in the columns' data type:

# %%
data.dtypes

# %% [markdown]
# If we look at the `"native-country"` column, we observe its data type is
# `object`, meaning it contains string values.
#
# ## Select features based on their data type
#
# In the previous notebook, we manually defined the numerical columns. We could
# do a similar approach. Instead, we can use the scikit-learn helper function
# `make_column_selector`, which allows us to select columns based on their data
# type. We now illustrate how to use this helper.

# %%
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns

# %% [markdown]
# Here, we created the selector by passing the data type to include; we then
# passed the input dataset to the selector object, which returned a list of
# column names that have the requested data type. We can now filter out the
# unwanted columns:

# %%
data_categorical = data[categorical_columns]
data_categorical

# %%
print(f"The dataset is composed of {data_categorical.shape[1]} features")

# %% [markdown]
# In the remainder of this section, we will present different strategies to
# encode categorical data into numerical data which can be used by a
# machine-learning algorithm.

# %% [markdown]
# ## Strategies to encode categories
#
# ### Encoding ordinal categories
#
# The most intuitive strategy is to encode each category with a different
# number. The `OrdinalEncoder` transforms the data in such manner. We start by
# encoding a single column to understand how the encoding works.

# %%
from sklearn.preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]

encoder = OrdinalEncoder().set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded

# %% [markdown]
# We see that each category in `"education"` has been replaced by a numeric
# value. We could check the mapping between the categories and the numerical
# values by checking the fitted attribute `categories_`.

# %%
encoder.categories_

# %% [markdown]
# Now, we can check the encoding applied on all categorical features.

# %%
data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]

# %%
print(f"The dataset encoded contains {data_encoded.shape[1]} features")

# %% [markdown]
# We see that the categories have been encoded for each feature (column)
# independently. We also note that the number of features before and after the
# encoding is the same.
#
# However, be careful when applying this encoding strategy:
# using this integer representation leads downstream predictive models
# to assume that the values are ordered (0 < 1 < 2 < 3... for instance).
#
# By default, `OrdinalEncoder` uses a lexicographical strategy to map string
# category labels to integers. This strategy is arbitrary and often
# meaningless. For instance, suppose the dataset has a categorical variable
# named `"size"` with categories such as "S", "M", "L", "XL". We would like the
# integer representation to respect the meaning of the sizes by mapping them to
# increasing integers such as `0, 1, 2, 3`.
# However, the lexicographical strategy used by default would map the labels
# "S", "M", "L", "XL" to 2, 1, 0, 3, by following the alphabetical order.
#
# The `OrdinalEncoder` class accepts a `categories` constructor argument to
# pass categories in the expected ordering explicitly. You can find more
# information in the
# [scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
# if needed.
#
# If a categorical variable does not carry any meaningful order information
# then this encoding might be misleading to downstream statistical models and
# you might consider using one-hot encoding instead (see below).
#
# ### Encoding nominal categories (without assuming any order)
#
# `OneHotEncoder` is an alternative encoder that prevents the downstream
# models to make a false assumption about the ordering of categories. For a
# given feature, it creates as many new columns as there are possible
# categories. For a given sample, the value of the column corresponding to the
# category is set to `1` while all the columns of the other categories
# are set to `0`.
#
# We can encode a single feature (e.g. `"education"`) to illustrate how the
# encoding works.

# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded

# %% [markdown]
# ```{note}
# `sparse_output=False` is used in the `OneHotEncoder` for didactic purposes,
# namely easier visualization of the data.
#
# Sparse matrices are efficient data structures when most of your matrix
# elements are zero. They won't be covered in detail in this course. If you
# want more details about them, you can look at
# [this](https://scipy-lectures.org/advanced/scipy_sparse/introduction.html#why-sparse-matrices).
# ```

# %% [markdown]
# We see that encoding a single feature gives a dataframe full of zeros
# and ones. Each category (unique value) became a column; the encoding
# returned, for each sample, a 1 to specify which category it belongs to.
#
# Let's apply this encoding on the full dataset.

# %%
print(f"The dataset is composed of {data_categorical.shape[1]} features")
data_categorical

# %%
data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]

# %%
print(f"The encoded dataset contains {data_encoded.shape[1]} features")

# %% [markdown]
# Look at how the `"workclass"` variable of the 3 first records has been encoded
# and compare this to the original string representation.
#
# The number of features after the encoding is more than 10 times larger than
# in the original data because some variables such as `occupation` and
# `native-country` have many possible categories.

# %% [markdown]
# ### Choosing an encoding strategy
#
# Choosing an encoding strategy depends on the underlying models and the type of
# categories (i.e. ordinal vs. nominal).

# %% [markdown]
# ```{note}
# In general `OneHotEncoder` is the encoding strategy used when the
# downstream models are **linear models** while `OrdinalEncoder` is often a
# good strategy with **tree-based models**.
# ```

# %% [markdown]
# Using an `OrdinalEncoder` outputs ordinal categories. This means
# that there is an order in the resulting categories (e.g. `0 < 1 < 2`). The
# impact of violating this ordering assumption is really dependent on the
# downstream models. Linear models would be impacted by misordered categories
# while tree-based models would not.
#
# You can still use an `OrdinalEncoder` with linear models but you need to be
# sure that:
# - the original categories (before encoding) have an ordering;
# - the encoded categories follow the same ordering than the original
#   categories.
# The **next exercise** highlights the issue of misusing `OrdinalEncoder` with
# a linear model.
#
# One-hot encoding categorical variables with high cardinality can cause
# computational inefficiency in tree-based models. Because of this, it is not
# recommended to use `OneHotEncoder` in such cases even if the original
# categories do not have a given order. We will show this in the **final
# exercise** of this sequence.

# %% [markdown]
# ## Evaluate our predictive pipeline
#
# We can now integrate this encoder inside a machine learning pipeline like we
# did with numerical data: let's train a linear classifier on the encoded data
# and check the generalization performance of this machine learning pipeline using
# cross-validation.
#
# Before we create the pipeline, we have to focus on the `native-country`.
# Let's recall some statistics regarding this column.

# %%
data["native-country"].value_counts()

# %% [markdown]
# We see that the `"Holand-Netherlands"` category is occurring rarely. This will
# be a problem during cross-validation: if the sample ends up in the test set
# during splitting then the classifier would not have seen the category during
# training and would not be able to encode it.
#
# In scikit-learn, there are some possible solutions to bypass this issue:
#
# * list all the possible categories and provide them to the encoder via the
#   keyword argument `categories` instead of letting the estimator automatically
#   determine them from the training data when calling fit;
# * set the parameter `handle_unknown="ignore"`, i.e. if an unknown category is
#   encountered during transform, the resulting one-hot encoded columns for this
#   feature will be all zeros;
# * adjust the `min_frequency` parameter to collapse the rarest categories
#   observed in the training data into a single one-hot encoded feature. If you
#   enable this option, you can also set `handle_unknown="infrequent_if_exist"`
#   to encode the unknown categories (categories only observed at predict time)
#   as ones in that last column.
#
# In this notebook we only explore the second option, namely
# `OneHotEncoder(handle_unknown="ignore")`. Feel free to evaluate the
# alternatives on your own, for instance using a sandbox notebook.

# %% [markdown]
# ```{tip}
# Be aware the `OrdinalEncoder` exposes a parameter also named `handle_unknown`.
# It can be set to `use_encoded_value`. If that option is chosen, you can define
# a fixed value that is assigned to all unknown categories during `transform`.
# For example, `OrdinalEncoder(handle_unknown='use_encoded_value',
# unknown_value=-1)` would set all values encountered during `transform` to `-1`
# which are not part of the data encountered during the `fit` call. You are
# going to use these parameters in the next exercise.
# ```

# %% [markdown]
# We can now create our machine learning pipeline.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
)

# %% [markdown]
# ```{note}
# Here, we need to increase the maximum number of iterations to obtain a fully
# converged `LogisticRegression` and silence a `ConvergenceWarning`. Contrary
# to the numerical features, the one-hot encoded categorical features are all
# on the same scale (values are 0 or 1), so they would not benefit from
# scaling. In this case, increasing `max_iter` is the right thing to do.
# ```

# %% [markdown]
# Finally, we can check the model's generalization performance only using the
# categorical columns.

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data_categorical, target)
cv_results

# %%
scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# As you can see, this representation of the categorical variables is slightly
# more predictive of the revenue than the numerical variables that we used
# previously. The reason being that we have more (predictive) categorical
# features than numerical ones.

# %% [markdown]
#
# In this notebook we have:
# * seen two common strategies for encoding categorical features: **ordinal
#   encoding** and **one-hot encoding**;
# * used a **pipeline** to use a **one-hot encoder** before fitting a logistic
#   regression.
