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
# # Case Study  - Titanic Survival

# %% [markdown]
# # Feature Extraction

# %% [markdown]
# Here we will talk about an important piece of machine learning: the extraction of
# quantitative features from data.  By the end of this section you will
#
# - Know how features are extracted from real-world data.
# - See an example of extracting numerical features from textual data
#
# In addition, we will go over several basic tools within scikit-learn which can be used to accomplish the above tasks.

# %% [markdown]
# ## What Are Features?

# %% [markdown]
# ### Numerical Features

# %% [markdown]
# Recall that data in scikit-learn is expected to be in two-dimensional arrays, of size
# **n_samples** $\times$ **n_features**.
#
# Previously, we looked at the iris dataset, which has 150 samples and 4 features

# %%
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data.shape)

# %% [markdown]
# These features are:
#
# - sepal length in cm
# - sepal width in cm
# - petal length in cm
# - petal width in cm
#
# Numerical features such as these are pretty straightforward: each sample contains a list
# of floating-point numbers corresponding to the features

# %% [markdown]
# ### Categorical Features

# %% [markdown]
# What if you have categorical features?  For example, imagine there is data on the color of each
# iris:
#
#     color in [red, blue, purple]
#
# You might be tempted to assign numbers to these features, i.e. *red=1, blue=2, purple=3*
# but in general **this is a bad idea**.  Estimators tend to operate under the assumption that
# numerical features lie on some continuous scale, so, for example, 1 and 2 are more alike
# than 1 and 3, and this is often not the case for categorical features.
#
# In fact, the example above is a subcategory of "categorical" features, namely, "nominal" features. Nominal features don't imply an order, whereas "ordinal" features are categorical features that do imply an order. An example of ordinal features would be T-shirt sizes, e.g., XL > L > M > S. 
#
# One work-around for parsing nominal features into a format that prevents the classification algorithm from asserting an order is the so-called one-hot encoding representation. Here, we give each category its own dimension.  
#
# The enriched iris feature set would hence be in this case:
#
# - sepal length in cm
# - sepal width in cm
# - petal length in cm
# - petal width in cm
# - color=purple (1.0 or 0.0)
# - color=blue (1.0 or 0.0)
# - color=red (1.0 or 0.0)
#
# Note that using many of these categorical features may result in data which is better
# represented as a **sparse matrix**, as we'll see with the text classification example
# below.

# %% [markdown]
# #### Using the DictVectorizer to encode categorical features

# %% [markdown]
# When the source data is encoded has a list of dicts where the values are either strings names for categories or numerical values, you can use the `DictVectorizer` class to compute the boolean expansion of the categorical features while leaving the numerical features unimpacted:

# %%
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

# %%
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
vec

# %%
vec.fit_transform(measurements).toarray()

# %%
vec.get_feature_names()

# %% [markdown]
# ### Derived Features

# %% [markdown]
# Another common feature type are **derived features**, where some pre-processing step is
# applied to the data to generate features that are somehow more informative.  Derived
# features may be based in **feature extraction** and **dimensionality reduction** (such as PCA or manifold learning),
# may be linear or nonlinear combinations of features (such as in polynomial regression),
# or may be some more sophisticated transform of the features.

# %% [markdown]
# ### Combining Numerical and Categorical Features

# %% [markdown]
# As an example of how to work with both categorical and numerical data, we will perform survival predicition for the passengers of the HMS Titanic.
#
# We will use a version of the Titanic (titanic3.xls) from [here](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls). We converted the .xls to .csv for easier manipulation but left the data is otherwise unchanged.
#
# We need to read in all the lines from the (titanic3.csv) file, set aside the keys from the first line, and find our labels (who survived or died) and data (attributes of that person). Let's look at the keys and some corresponding example lines.

# %%
import os
import pandas as pd

titanic = pd.read_csv(os.path.join('datasets', 'titanic3.csv'))
print(titanic.columns)

# %% [markdown]
# Here is a broad description of the keys and what they mean:
#
# ```
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# survival        Survival
#                 (0 = No; 1 = Yes)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat            Lifeboat
# body            Body Identification Number
# home.dest       Home/Destination
# ```
#
# In general, it looks like `name`, `sex`, `cabin`, `embarked`, `boat`, `body`, and `homedest` may be candidates for categorical features, while the rest appear to be numerical features. We can also look at the first couple of rows in the dataset to get a better understanding:

# %%
titanic.head()

# %% [markdown]
# We clearly want to discard the "boat" and "body" columns for any classification into survived vs not survived as they already contain this information. The name is unique to each person (probably) and also non-informative. For a first try, we will use "pclass", "sibsp", "parch", "fare" and "embarked" as our features:

# %%
labels = titanic.survived.values
features = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# %%
features.head()

# %% [markdown]
# The data now contains only useful features, but they are not in a format that the machine learning algorithms can understand. We need to transform the strings "male" and "female" into binary variables that indicate the gender, and similarly for "embarked".
# We can do that using the pandas ``get_dummies`` function:

# %%
pd.get_dummies(features).head()

# %% [markdown]
# This transformation successfully encoded the string columns. However, one might argue that the class is also a categorical variable. We can explicitly list the columns to encode using the ``columns`` parameter, and include ``pclass``:

# %%
features_dummies = pd.get_dummies(features, columns=['pclass', 'sex', 'embarked'])
features_dummies.head(n=16)

# %%
data = features_dummies.values

# %%
import numpy as np
np.isnan(data).any()

# %% [markdown]
# With all of the hard data loading work out of the way, evaluating a classifier on this data becomes straightforward. Setting up the simplest possible model, we want to see what the simplest score can be with `DummyClassifier`.

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, random_state=0)

imp = Imputer()
imp.fit(train_data)
train_data_finite = imp.transform(train_data)
test_data_finite = imp.transform(test_data)

# %%
np.isnan(train_data_finite).any()

# %%
from sklearn.dummy import DummyClassifier

clf = DummyClassifier('most_frequent')
clf.fit(train_data_finite, train_labels)
print("Prediction accuracy: %f"
      % clf.score(test_data_finite, test_labels))

# %% [markdown]
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Try executing the above classification, using LogisticRegression and RandomForestClassifier instead of DummyClassifier
#       </li>
#       <li>
#       Does selecting a different subset of features help?
#       </li>
#     </ul>
# </div>

# %% {"deletable": true, "editable": true}
# # %load solutions/10_titanic.py
