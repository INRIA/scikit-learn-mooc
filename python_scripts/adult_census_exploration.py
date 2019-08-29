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
# In this notebook, we will look at necessary steps that happen before any machine learning takes place. 
# * load the data
# * look at the variables in the dataset, in particular make the difference
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset.

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns
sns.set_context('talk')

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv('datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)


# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].unique()

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for exemple `native-country`.

# %%
numerical_columns = ['age', 'education-num', 'capital-gain', 'capital-loss',
                     'hours-per-week']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Visualize the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise in the data collection (for example, malfunctioning sensor or missing
#   values), or the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. `pandas_profiling` is a nice tool for this.

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# TODO: some comments about a few variables?
# * age: retired people are not in the dataset (`hours-per-week > 0`).
#
# * education num: peak at TODO and TODO probably correspond to under-graduate and masters?
# * hours per week around 40, this was probably the standard at the time
#
# TODO: show categorical variables distribution maybe?

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how variable
# differ according to the class. In the diagonal you can see the distribution
# of individual variables. The plots on the off-diagonal can reveal interesting
# interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
sns.pairplot(data=adult_census[:n_samples_to_plot] , vars=columns,
             hue=target_column, plot_kws={'alpha': 0.2}, height=4,
             diag_kind='hist');

# TODO talk about a few expected things that make sense:
# * hours per week around 40, high-revenue work more, low-revenue part-time work
# * education-num peak at TODO and TODO probably for undergraduate and masters?? low-revenue tail on the lhs
# * age: young people have low-revenue.
# * classes are slighly imbalanced. Class imbalance happens often in practice
#   and may need special techniques for machine learning. For example in a
#   medical setting, there are a lot less patients with a rare disease than sane
#   patients.
# %%
sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age', y_vars='hours-per-week',
             hue=target_column, markers=['o', 'v'], plot_kws={'alpha': 0.2}, height=12);

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-writen rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) you tend to work less and if you are old (more than 70
# year-old roughly). This is a non-linear relationship between age and hours
# per week. Some machine learning models can capture only linear interaction so
# this may be important when deciding which model to chose.


# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and make the difference
#   between categorical and numerical variables.
# * inspected the data with `pandas_profiling` and `seaborn`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to notice potential peculiarities in your data.
