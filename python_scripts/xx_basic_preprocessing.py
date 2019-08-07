# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to scikit-learn: basic preprocessing for basic model fitting
#
# In this lecture note, we will aim at introducing:
# * the difference between numerical and categorical variables;
# * the importance of scaling numerical variables;
# * the way to encode categorical variables;
# * combine different preprocessing on different type of data;
# * evaluate the performance of a model via cross-validation.
#
# ## Introduce the dataset
#
# To this aim, we will use data from the 1985 "Current Population Survey"
# (CPS). The goal with this data is to regress wages from heterogeneous data
# such as age, experience, education, family information, etc.
#
# Let's first load the data located in the `datasets` folder.

# %%
import os
import time
import pandas as pd

df = pd.read_csv(os.path.join('datasets', 'cps_85_wages.csv'))

# %% [markdown]
# We can quickly have a look at the head of the dataframe to check the type
# of available data.

# %%
print(df.head())

# %% [markdown]
# The target in our study will be the "WAGE" columns while we will use the
# other columns to fit a model

# %%
target_name = "WAGE"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

# %% [markdown]
# We can check the number of samples and the number of features available in
# the dataset

# %%
print(
    f"The dataset contains {data.shape[0]} samples and {data.shape[1]} "
    "features"
)

# %% [markdown]
# ## Work with numerical data
#
# The most intuitive type of data in machine learning which can (almost)
# directly be used in machine learning are known as numerical data. We can
# quickly have a look at such data by selecting the subset of columns from
# the original data.

# %%
print(data.columns)
numerical_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']

# %% [markdown]
# We will use this subset of data to fit linear regressor to infer the wage

# %%
data_numeric = data[numerical_columns]

# %% [markdown]
# When building a machine learning model, it is important to leave out a
# subset of the data which we can use later to evaluate the trained model.
# The data used to fit a model a called training data while the one used to
# assess a model are called testing data.
#
# Scikit-learn provides an helper function `train_test_split` which will
# split the dataset into a training and a testing set. It will ensure that
# the data are shuffled before splitting the data.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42
)

print(
    f"The training dataset contains {data_train.shape[0]} samples and "
    f"{data_train.shape[1]} features"
)
print(
    f"The testing dataset contains {data_test.shape[0]} samples and "
    f"{data_test.shape[1]} features"
)

# %% [markdown]
# We will build a Support Vector Machine (SVM) which is a linear model. The
# `fit` method is called to train the data and only the training data should
# be given for this purpose.
# To evaluate our model, we can use the method `score`. It will compute the
# coefficient of determination R2 when dealing with a regression problem.
#
# In addition, we checking the time required to train the model and internally
# check the number of iterations done by the solver to find a solution.
# %%
from sklearn.svm import LinearSVR

model = LinearSVR()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations"
)

# %% [markdown]
# We should not the `ConvergenceWarning` which inform us that our model stopped
# learning since it reaches the maximum number of iterations allowed by the
# user. This could potentially be detrimental for the model accuracy. We can
# follow the (bad) advice given in the warning message and increase the maximum
# number of iterations allowed.

# %%
model = LinearSVR(max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations"
)

# %% [markdown]
# We can observe an increase in performance add the cost of a longer training.
# Instead of increasing the number of iterations, we could instead know a bit
# more about the SVR model and known that it is expecting input data to be
# scaled before to start training. A range of preprocessing algorithms in
# scikit-learn allows to transform the input data before to train a model.
# We can easily combine these sequential operation with a scikit-learn
# `Pipeline` which will chain the operations and can be used as any other
# classifier or regressor. The helper function `make_pipeline` will create
# a `Pipeline` by giving the successive transformations to perform.
#
# In our case, we will standardize the data and then train a linear SVR.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LinearSVR())
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model[-1].n_iter_} iterations"
)

# %% [markdown]
# We can see that the training time and the number of iterations is much
# shorter while the accuracy is equivalent.
#
# In the previous example, we split the original data into a training set and
# a testing set. This strategy has several issues: in the setting where the
# amount of data is limited, the subset of data used to train or test will be
# small; and the splitting was done in a random manner and we have no
# information regarding the confidence of the results obtained.
#
# Therefore, we can use what cross-validation. Cross-validation consists in
# repeating this random splitting into training and testing sets and aggregate
# the model performance. By repeating the experiment, one can get a the
# fluctuation of the model performance.
#
# The function `cross_val_score` allows for such experimental protocol by
# giving the model, the data and the target. Since there is several
# cross-validation strategies, `cross_val_score` takes a parameter `cv` which
# defines the splitting strategy.

# %%
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, data_numeric, target, cv=5)
print(f"The R2 score (mean +/- 1 std. dev.) is: "
      f"{score.mean():.2f} +/- {score.std():.2f}")
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# Setting `cv=5` created 5 training and testing sets on which we trained and
# tested a model. This strategy is called K-fold cross-validation where `K`
# corresponds to the number of split.
#
# ## Work with categorical data
#
# In the previous section, we dealt with data for which numerical algorithms
# are mathematically designed to work natively. However, real datasets contain
# type of data which do not belong to this category and will require some
# preprocessing. Such preprocessing will transform these data to be numerical
# and thus natively handled by machine learning algorithms.
#
# Categorical data are broadly encountered in data science. Numerical data is a
# continuous quantity corresponding to a real numbers while categorical data
# are represented as discrete values. For instance, the variable `SEX` in our
# previous dataset is a categorical variable because it encodes the data with
# the two categories `male` and `female`.
#
# In the remainder of this section, we will present different strategies to
# encode categorical data into numerical data which can be used by a
# machine-learning algorithm.

# %%
categorical_columns = [
    'SOUTH', 'SEX', 'UNION', 'RACE', 'OCCUPATION', 'SECTOR', 'MARR'
]
data_categorical = data[categorical_columns]
print(data_categorical.head())

# %% [markdown]
# ### Encode categories having an ordering
#
# The most intuitive strategy is to encode each category by a numerical value.
# The `OrdinalEncoder` will transform the data in such manner.

# %%
from sklearn.preprocessing import OrdinalEncoder

print(data_categorical.head())
print(f"The datasets is composed of {data_categorical.shape[1]} features")
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
print(data_encoded[:5])

# %% [markdown]
# We can see that all categories have been encoded for each feature
# independently. We can also notice that the number of feature before and after
# the encoding is the same.
#
# However, one has to be careful when using this encoding strategy. The
# encoding imposed an order regarding the categories: 0 is smaller than 1 which
# is smaller than 2, etc. If the original categories did not have such order
# then this encoding is not adequate and you should use one-hot encoding
# instead.
#
# ### Encode categories which do not have an ordering
#
# As previously stated, `OrdinalEncoder` is encoding categorical data having
# an ordering. In this case, the `OneHotEncoder` should be used. For a given
# feature, it will create as many new columns as categories. For a sample,
# the column corresponding to the category will be set to `1` while the other
# columns will be set to `0`.

# %%
from sklearn.preprocessing import OneHotEncoder

print(data_categorical.head())
print(f"The datasets is composed of {data_categorical.shape[1]} features")
encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
print(data_encoded[:5])

# %% [markdown]
# One can notice that the number of features after the encoding is larger than
# in the original data.
#
# Once the encoding is done, we could integrate it inside a machine learning
# pipeline as in the case with numerical data. In the following, we train a
# linear classifier on the encoded data and check the performance of this
# machine learning pipeline using cross-validation.

# %%
model = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LinearSVR())
score = cross_val_score(model, data_categorical, target)
print(f"The R2 score is: {score.mean():.2f} +/- {score.std():.2f}")
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# ## Combining different preprocessing on different data type
#
# In the previous section, we saw that we need to treat data specifically
# depending of their nature (i.e. numerical or categorical). We were capable
# of making the preprocessing in two sequential steps but we did not present
# any tool which could allow us to first preprocess the data depending of
# their type and later on use these preprocessed data to train a single
# machine learning model.
#
# Scikit-learn provides a `ColumnTransformer` which will dispatch some specific
# columns to a specific transformer.
#
# We can first define the columns depending on their data type:
# * binary encoding: it will corresponds to features for which categories can
#   be encoded by the values 0 or 1.
# * one-hot encoding: it will corresponds to features for which categories
#   do not have a particular ordered and should be a column should be created
#   for each category.
# * scaling: it will corresponds to the numerical features which will be
#   standardized.

# %%
binary_encoding_columns = ['MARR', 'SEX', 'SOUTH', 'UNION']
one_hot_encoding_columns = ['OCCUPATION', 'SECTOR', 'RACE']
scaling_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']

# %% [markdown]
# We can now create our `ColumnTransfomer` by specifying a list of triplet
# (preprocessor name, transformer, columns). Finally, we can merge this
# "preprocessor" in a machine learning pipeline by adding a machine learning
# model (e.g. a linear model) after the preprocessing.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])
model = make_pipeline(preprocessor, RidgeCV())
score = cross_val_score(model, data, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(f"The different scores obtained are: \n{score}")


# %% [markdown]
# One can notice that the model, even more complex than in the previous
# sections, follow the same API meaning that it `fit` is called to preprocess
# the data and `score` is used to predict and check the model performance.
