# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,python_scripts//py:percent
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
# # Introduction to scikit-learn
#
# ## Basic preprocessing and model fitting
#
# In this notebook, we present how to build predictive models on tabular
# datasets.
#
# In particular we will highlight:
# * the difference between numerical and categorical variables;
# * the importance of scaling numerical variables;
# * typical ways to deal categorical variables;
# * train predictive models on different kinds of data;
# * evaluate the performance of a model via cross-validation.
#
# ## Introducing the dataset
#
# To this aim, we will use data from the 1994 Census bureau database. The goal
# with this data is to regress wages from heterogeneous data such as age,
# employment, education, family information, etc.
#
# Let's first load the data located in the `datasets` folder.

# %%
import pandas as pd

df = pd.read_csv("https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')


# %% [markdown]
# Let's have a look at the first records of this data frame:

# %%
df.head()


# %% [markdown]
# The target variable in our study will be the "class" column while we will use
# the other columns as input variables for our model. This target column divides
# the samples (also known as records) into two groups: high income (>50K) vs low
# income (<=50K). The resulting prediction problem is therefore a binary
# classification problem.
#
# For simplicity, we will ignore the "fnlwgt" (final weight) column that was
# crafted by the creators of the dataset when sampling the dataset to be
# representative of the full census database.

# %%
target_name = "class"
target = df[target_name].to_numpy()
target


# %%
data = df.drop(columns=[target_name, "fnlwgt"])
data.head()

# %% [markdown]
# We can check the number of samples and the number of features available in
# the dataset:

# %%
print(
    f"The dataset contains {data.shape[0]} samples and {data.shape[1]} "
    "features"
)


# %% [markdown]
# ## Working with numerical data
#
# The numerical data is the most natural type of data used in machine learning
# and can (almost) directly be fed to predictive models. We can quickly have a
# look at such data by selecting the subset of columns from the original data.
#
# We will use this subset of data to fit a linear classification model to
# predict the income class.

# %%
data.columns


# %%
data.dtypes


# %%
numerical_columns = [c for c in data.columns
                     if data[c].dtype.kind in ["i", "f"]]
numerical_columns

# %%
data_numeric = data[numerical_columns]
data_numeric.head()

# %% [markdown]
# When building a machine learning model, it is important to leave out a
# subset of the data which we can use later to evaluate the trained model.
# The data used to fit a model a called training data while the one used to
# assess a model are called testing data.
#
# Scikit-learn provides an helper function `train_test_split` which will
# split the dataset into a training and a testing set. It will ensure that
# the data are shuffled randomly before splitting the data.

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
# We will build a linear classification model called "Logistic Regression". The
# `fit` method is called to train the model from the input and target data. Only
# the training data should be given for this purpose.
#
# In addition, when checking the time required to train the model and internally
# check the number of iterations done by the solver to find a solution.
# %%
from sklearn.linear_model import LogisticRegression
import time

model = LogisticRegression()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

print(
    f"The model {model.__class__.__name__} was trained in "
    f"{elapsed_time:.3f} seconds for {model.n_iter_} iterations"
)


# %% [markdown]
# Let's ignore the convergence warning for now and instead let's try
# to use our model to make some predictions on the first three records
# of the held out test set:

# %%
target_predicted = model.predict(data_test)
target_predicted[:5]


# %%
target_test[:5]


# %%
predictions = data_test.copy()
predictions['predicted-class'] = target_predicted
predictions['expected-class'] = target_test
predictions['correct'] = target_predicted == target_test
predictions.head()


# %% [markdown]
# To quantitatively evaluate our model, we can use the method `score`. It will
# compute the classification accuracy when dealing with a classificiation
# problem.

# %%
print(
    f"The test accuracy using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.3f}"
)


# %% [markdown]
# This is mathematically equivalent as computing the average number of time
# the model makes a correct prediction on the test set:

# %%
(target_test == target_predicted).mean()


# %% [markdown]
# Let's now consider the `ConvergenceWarning` message that was raised previously
# when calling the `fit` method to train our model. This warning informs us that
# our model stopped learning becaused it reached the maximum number of
# iterations allowed by the user. This could potentially be detrimental for the
# model accuracy. We can follow the (bad) advice given in the warning message
# and increase the maximum number of iterations allowed.

# %%
model = LogisticRegression(max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The accuracy using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.3f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations"
)


# %% [markdown]
# We can observe now a longer training time but not significant improvement in
# the predictive performance. Instead of increasing the number of iterations, we
# can try to help fit the model faster by scaling the data first. A range of
# preprocessing algorithms in scikit-learn allows to transform the input data
# before training a model. We can easily combine these sequential operation with
# a scikit-learn `Pipeline` which will chain the operations and can be used as
# any other classifier or regressor. The helper function `make_pipeline` will
# create a `Pipeline` by giving the successive transformations to perform.
#
# In our case, we will standardize the data and then train a new logistic
# regression model on that new version of the dataset set.

# %%
data_train.describe()


# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled

# %%
data_train_scaled = pd.DataFrame(data_train_scaled, columns=data_train.columns)
data_train_scaled.describe()


# %%
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The accuracy using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.3f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model[-1].n_iter_} iterations"
)


# %% [markdown]
# We can see that the training time and the number of iterations is much shorter
# while the predictive performance (accuracy) stays the same.
#
# In the previous example, we split the original data into a training set and a
# testing set. This strategy has several issues: in the setting where the amount
# of data is limited, the subset of data used to train or test will be small;
# and the splitting was done in a random manner and we have no information
# regarding the confidence of the results obtained.
#
# Instead, we can use what cross-validation. Cross-validation consists in
# repeating this random splitting into training and testing sets and aggregate
# the model performance. By repeating the experiment, one can get an estimate of
# the variabilty of the model performance.
#
# The function `cross_val_score` allows for such experimental protocol by giving
# the model, the data and the target. Since there exists several
# cross-validation strategies, `cross_val_score` takes a parameter `cv` which
# defines the splitting strategy.
#
#
#
#
#
#
#


# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data_numeric, target, cv=5)
print(f"The different scores obtained are: \n{scores}")


# %%
print(f"The accuracy (mean +/- 1 std. dev.) is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# Note that by computing the standard-deviation of the cross-validation scores
# we can get an idea of the uncertainty of our estimation of the predictive
# performance of the model: in the above results, only the first 2 decimals seem
# to be trustworthy. Using a single train / test split would not allow us to
# know anything about the level of uncertainty of the accuracy of the model.
#
# Setting `cv=5` created 5 distinct splits to get 5 variations for the training
# and testing sets. Each training set is used to fit one model which is then
# scored on the matching test set. This strategy is called K-fold
# cross-validation where `K` corresponds to the number of splits.
#
# The following matplotlib code helps visualize how the datasets is partitionned
# between train and test samples at each iteration of the cross-validation
# procedure:

# %%
# %matplotlib inline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

cmap_cv = plt.cm.coolwarm

def plot_cv_indices(cv, X, y, ax, lw=20):
    """Create a sample plot for indices of a cross-validation object."""
    splits = list(cv.split(X=X, y=y))
    n_splits = len(splits)

    # Generate the training/testing visualizations for each CV split
    for ii, (train, test) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.zeros(shape=X.shape[0], dtype=np.int32)
        indices[train] = 1

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + .2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


# %%
# Some random data points
n_points = 100
X = np.random.randn(n_points, 10)
y = np.random.randn(n_points)

fig, ax = plt.subplots(figsize=(10, 6))
cv = KFold(5)
plot_cv_indices(cv, X, y, ax);

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
model = make_pipeline(OneHotEncoder(handle_unknown='ignore'),
                      LogisticRegression(max_iter=1000))
scores = cross_val_score(model, data_categorical, target)
print(f"The different scores obtained are: \n{scores}")


# %%
print(f"The accuracy is: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# As you can see, this representation of the categorical variables of the data is slightly more predictive of the revenue than the numerical variables that we used previously.

# %% [markdown]
# ## Exercise:
# - Try to use the OrdinalEncoder instead. What do you observe?
#
# In case you have issues of with unknown categories, try to precompute the list
# of possible categories ahead of time and pass it explicitly to the constructor
# of the encoder:
#
# ```python
# categories = [data[column].unique()
#               for column in data[categorical_columns]]
# OrdinalEncoder(categories=categories)
# ```


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
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])
model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

# %% [markdown]
# The final model is more complex than the previous models but still follows the
# same API:
# - the `fit` method is called to preprocess the data then train the classifier;
# - the `predict` method can make predictions on new data;
# - the `score` method is used to predict on the test data and compare the
#   predictions to the expected test labels to compute the accuracy.

# %%
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
# ## Exercises:
#
# - Check that scaling the numerical features does not impact the speed or
#   accuracy of HistGradientBoostingClassifier
# - Check that one-hot encoding the categorical variable does not improve the
#   accuracy of HistGradientBoostingClassifier but slows down the training.

# %%
