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

model = LogisticRegression(solver='lbfgs')
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
# ## Exercise 1
#
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <= 50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
#
# Hint: You can compute the cross-validated of a [DummyClassifier](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) the performance of such baselines.
#
# Use the dedicated notebook to do this exercise.

# %% [markdown]
# Let's now consider the `ConvergenceWarning` message that was raised previously
# when calling the `fit` method to train our model. This warning informs us that
# our model stopped learning becaused it reached the maximum number of
# iterations allowed by the user. This could potentially be detrimental for the
# model accuracy. We can follow the (bad) advice given in the warning message
# and increase the maximum number of iterations allowed.

# %%
model = LogisticRegression(solver='lbfgs', max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start


# %%
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

model = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start


# %%
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
print(f"The mean cross-validation accuracy is: "
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

# TODO: add summary here
