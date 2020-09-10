# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Loading data into machine learning
#
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to learn more about this dataset: http://www.openml.org/d/1590

# The goal with this data is to predict whether a person earns over 50K a year
# from heterogeneous data such as age, employment, education, family
# information, etc.

# %% [markdown]
# ## The variables (columns) in the dataset
#
# The data are stored in a pandas dataframe.
#
# Pandas is a Python library to manipulate tables, a bit like Excel but programming: https://pandas.pydata.org/

# %%
adult_census.head()  # Look at the first few lines of our dataframe

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue). The resulting prediction problem is therefore a binary
# classification problem,
# while we will use the other columns as input variables for our model.

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
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
# We can check the number of samples and the number of features available in
# the dataset:

# %%
print(
    f"The dataset contains {adult_census.shape[0]} samples and "
    f"{adult_census.shape[1]} features")

# %% [markdown]
# ## Visual inspection of the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the information you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education.
#
# Let's look at the relationship between `education` and `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']

# reset the plotting style
import matplotlib.pyplot as plt
plt.rcdefaults()

import seaborn as sns
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=3, diag_kind='hist')

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
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
# less. This is a non-linear relationship between age and hours
# per week. Linear machine learning models can only capture linear interactions, so
# this may be a factor when deciding which model to chose.
#
# In a machine-learning setting, algorithm automatically
# decide what should be the "rules" in order to make predictions on new data.
# Let's visualize which set of simple rules a decision tree would grasp using the
# same data.


# %%
def plot_tree_decision_function(tree, X, y, ax=None):
    """Plot the different decision rules found by a `DecisionTreeClassifier`.

    Parameters
    ----------
    tree : DecisionTreeClassifier instance
        The decision tree to inspect.
    X : dataframe of shape (n_samples, n_features)
        The data used to train the `tree` estimator.
    y : ndarray of shape (n_samples,)
        The target used to train the `tree` estimator.
    ax : matplotlib axis
        The matplotlib axis where to plot the different decision rules.
    """
    import numpy as np
    from scipy import ndimage

    h = 0.02
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    faces = tree.tree_.apply(
        np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    faces = faces.reshape(xx.shape)
    border = ndimage.laplace(faces) != 0
    if ax is None:
        ax = plt.gca()
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1],
               c=np.array(['tab:blue',
                           'tab:red'])[y], s=60, alpha=0.7)
    ax.contourf(xx, yy, Z, alpha=.4, cmap='RdBu_r')
    ax.scatter(xx[border], yy[border], marker='.', s=1)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    sns.despine(offset=10)


# %%
from sklearn.preprocessing import LabelEncoder

# select a subset of data
data_subset = adult_census[:n_samples_to_plot]
X = data_subset[["age", "hours-per-week"]]
y = LabelEncoder().fit_transform(
    data_subset[target_column].to_numpy())

# %% [markdown]
# We will create a simple decision tree with a maximum of 2 rules, in order
# to interpret the results.

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

max_leaf_nodes = 3
tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                              random_state=0)
tree.fit(X, y)

# %% [markdown]
# `plot_tree` will allow us to visually check the set of rules learnt by
# the decision tree.

# %%
_ = plot_tree(tree)

# %%
# plot the decision function learned by the tree
plot_tree_decision_function(tree, X, y)

# %% [markdown]
# By allowing only 3 leaves in the tree, we get similar rules to the ones we
# designed by hand:
# * the persons younger than 28.5 year-old (X[0] < 28.5) will be considered in the class
#   earning `<= 50K`.
# * the persons older than 28.5 and working less than 40.5 hours-per-week (X[1] <= 40.5)
#   will be considered in the class earning `<= 50K`, while the persons working
#   above 40.5 hours-per-week, will be considered in the class
#   earning `> 50K`.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the differents kind of variables to differentiate
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Ideas which will be discussed more in details later:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for training and
#   evaluating your machine learning model
# * having redundant (or highly correlated) columns can be a problem for
#   some machine learning algorithms
# * contrary to decision tree, linear models can only capture linear interaction, so be
#   aware of non-linear relationships in your data
