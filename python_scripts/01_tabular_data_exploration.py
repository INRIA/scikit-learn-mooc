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
# # First look at our dataset
#
# In this notebook, we will look at the necessary steps required before any
#  machine learning takes place. It involves:
#
# * loading the data;
# * looking at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows;
# * visualizing the distribution of the variables to gain some insights into
#   the dataset.

# %% [markdown]
# ## Loading the adult census dataset
#
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).
#
# We use pandas to read this dataset.
#
# ```{note}
# [Pandas](https://pandas.pydata.org/) is a Python library used for
# manipulating 1 and 2 dimensional structured data.
# ```

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We can look at the OpenML webpage to learn more about this dataset:
# <http://www.openml.org/d/1590>
#
# The goal with this data is to predict whether a person earns over 50K a year
# from heterogeneous data such as age, employment, education, family
# information, etc.

# %% [markdown]
# ## The variables (columns) in the dataset
#
# The data are stored in a pandas dataframe. A dataframe is type of structured
# data composed of 2 dimensions. This type of data are also referred as tabular
# data.
#
# The rows represents a record. In the field of machine learning or descriptive
# statistics, the terms commonly used to refer to rows are "sample",
# "instance", or "observation".
#
# The columns represents a type of information collected. In the field of
# machined learning and descriptive statistics, the terms commonly used to
# refer to columns are "feature", "variable", "attribute", or "covariate".

# %%
adult_census.head()  # Print the first few lines of our dataframe

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<=50K` (low-revenue) and
# `>50K` (high-revenue). The resulting prediction problem is therefore a
# binary classification problem, while we will use the other columns as input
# variables for our model.

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# ```{note}
# Classes are slightly imbalanced, meaning there are more samples of one or
# more classes compared to others. Class imbalance happens often in practice
# and may need special techniques when building a predictive model.
#
# For example in a medical setting, if we are trying to predict whether
# subjects will develop a rare disease, there will be a lot more healthy
# subjects than ill subjects in the dataset.
# ```

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# take continuous values, for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# We can check the number of samples and the number of columns available in
# the dataset:

# %%
print(f"The dataset contains {adult_census.shape[0]} samples and "
      f"{adult_census.shape[1]} columns")

# %% [markdown]
# We can compute the number of features by counting the number of columns and
# subtract 1, since of the column is the target.

# %%
print(f"The dataset contains {adult_census.shape[1] - 1} features.")

# %% [markdown]
# ## Visual inspection of the data
# Before building a predictive model, it is a good idea to look at the data:
#
# * maybe the task you are trying to achieve can be solved without machine
#   learning;
# * you need to check that the information you need for your task is actually
#   present in the dataset;
# * inspecting the data is a good way to find peculiarities. These can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example
#   capped values).

# %% [markdown]
# Let's look at the distribution of individual features, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for features containing numerical values:

# %%
_ = adult_census.hist(figsize=(20, 14))

# %% [markdown]
# ```{tip}
# In the cell, we are calling the following pattern: `_ = func()`. It assigns
# the output of `func()` into the variable called `_`. By convention, in Python
# `_` serves as a "garbage" variable to store results that we are not
# interested in.
# ```
#
# We can already make a few comments about some of the variables:
#
# * `age`: there are not that many points for 'age > 70'. The dataset
#   description does indicate that retired people have been filtered out
#   (`hours-per-week > 0`);
# * `education-num`: peak at 10 and 13, hard to tell what it corresponds to
#   without looking much further. We'll do that later in this notebook;
# * `hours-per-week` peaks at 40, this was very likely the standard number of
#   working hours at the time of the data collection;
# * most values of `capital-gain` and `capital-loss` are close to zero.

# %% [markdown]
# For categorical variables, we can look at the distribution of values:

# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number
# of years of education.
#
# Let's look at the relationship between `education` and `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that `education` and `education-num` gives you the same
# information. For example, `education-num=2` is equivalent to
# `education='1st-4th'`. In practice that means we can remove `education-num`
# without losing information. Note that having redundant (or highly correlated)
# columns can be a problem for machine learning algorithms.

# %% [markdown]
# ```{note}
# In the upcoming notebooks, we will only keep the `education` variable,
# excluding the `education-num` variable.
# ```

# %% [markdown]
# Another way to inspect the data is to do a `pairplot` and show how each
# variable differs according to our target, `class`. Plots along the diagonal
# show the distribution of individual variables for each `class`. The plots on
# the off-diagonal can reveal interesting interactions between variables.

# %%
import seaborn as sns

n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=3, diag_kind='hist', diag_kws={'bins': 30})

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
#
# * if you are young (less than 25 year-old roughly), you are in the
#   `<=50K` class;
# * if you are old (more than 70 year-old roughly), you are in the
#   `<=50K` class;
# * if you work part-time (less than 40 hours roughly) you are in the
#   `<=50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours per week.
# Linear machine learning models can only capture linear interactions, so this
# may be a factor when deciding which model to chose.
#
# In a machine-learning setting, an algorithm automatically create the "rules"
# in order to make predictions on new data.

# %% [markdown]
# The plot below shows the rules of a simple model, called decision tree.
# We will explain how this model works in a latter notebook, for now let us
# just consider the model predictions when trained on this dataset:
#
# ![](../figures/simple_decision_tree_adult_census.png)
#
# The background color in each area represents the probability of the class
# `high-income` as estimated by the model. Values towards 0 (dark blue)
# indicates that the model predicts `low-income` with a high probability.
# Values towards 1 (dark orange) indicates that the model predicts
# `high-income` with a high probability. Values towards 0.5 (white) indicates
# that the model is not very sure about its prediction.
#
# Looking at the plot here is what we can gather:
#
# * In the region `age < 28.5` (left region) the prediction is `low-income`.
#   The dark blue color indicates that the model is quite sure about its
#   prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5`
#   (bottom-right region), the prediction is `low-income`. Note that the blue
#   is a bit lighter that for the left region which means that the algorithm is
#   not as certain in this region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. However the probability of the class
#   `low-income` is very close to 0.5 which means the model is not sure at all
#   about its prediction.
#
# It is interesting to see that a simple model create rules similar to the ones
# that we could have created by hand. Note that machine learning is really
# interesting when creating rules by hand is not straightforward, for example
# because we are in high dimension (many features) or because there is no
# simple and obvious rules that separate the two classes as in the top-right
# region

# %% [markdown]
#
# In this notebook we have:
#
# * loaded the data from a CSV file using `pandas`;
# * looked at the different kind of variables to differentiate between
#   categorical and numerical variables;
# * inspected the data with `pandas` and `seaborn`. Data inspection can allow
#   you to decide whether using machine learning is appropriate for your data
#   and to highlight potential peculiarities in your data.
#
# Ideas which will be discussed more in details later:
#
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for training
#   and evaluating your machine learning model;
# * having redundant (or highly correlated) columns can be a problem for
#   some machine learning algorithms;
# * contrary to decision tree, linear models can only capture linear
#   interaction, so be aware of non-linear relationships in your data.
