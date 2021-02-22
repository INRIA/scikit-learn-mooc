# %% [markdown]
# # Working with numerical data
#
# In the previous notebook, we trained a k-nearest neighbors model on
# some data.
#
# However, we oversimplified the procedure by loading a dataset that contained
# exclusively numerical data. Besides, we used datasets which were already
# split into train-test sets.
#
# In this notebook, we aim at:
#
# * identifying numerical data in a heterogeneous dataset;
# * selecting the subset of columns corresponding to numerical data;
# * using a scikit-learn helper to separate data into train-test sets;
# * training and evaluating a more complex scikit-learn model.
#
# We will start by loading the adult census dataset used during the data
# exploration.
#
# ## Loading the entire dataset
#
# As in the previous notebook, we rely on pandas to open the CSV file into
# a pandas dataframe.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
adult_census.head()

# %% [markdown]
# The next step separates the target from the data. We performed the same
# procedure in the previous notebook.

# %%
data, target = adult_census.drop(columns="class"), adult_census["class"]

# %%
data.head()

# %%
target

# %% [markdown]
# ```{caution}
# Here and later, we use the name `data` and `target` to be explicit. In
# scikit-learn, documentation `data` is commonly named `X` and `target` is
# commonly called `y`.
# ```

# %% [markdown]
# At this point, we can focus on the data we want to use to train our
# predictive model.
#
# ## Identify numerical data
#
# Numerical data are represented with numbers. They are linked to measurable
# (quantitative) data, such as age or the number of hours a person works a
# week.
#
# Predictive models are natively designed to work with numerical data.
# Moreover, numerical data usually requires very little work before getting
# started with training.
#
# The first task here will be to identify numerical data in our dataset.
#
# ```{caution}
# Numerical data are represented with numbers, but numbers are not always
# representing numerical data. Categories could already be encoded with
# numbers and you will need to identify these features.
# ```
#
# Thus, we can check the data type for each of the column in the dataset.

# %%
data.dtypes

# %% [markdown]
# We seem to have only two data types. We can make sure by checking the unique
# data types.

# %%
data.dtypes.unique()

# %% [markdown]
# Indeed, the only two types in the dataset are integer and object.
# We can look at the first few lines of the dataframe to understand the
# meaning of the `object` data type.

# %%
data.head()

# %% [markdown]
# We see that the `object` data type corresponds to columns containing strings.
# As we saw in the exploration section, these columns contain categories and we
# will see later how to handle those. We can select the columns containing
# integers and check their content.

# %%
numerical_columns = [
    "age", "education-num", "capital-gain", "capital-loss",
    "hours-per-week", "fnlwgt"]
data[numerical_columns].head()

# %% [markdown]
# Now that we limited the dataset to numerical columns only,
# we can analyse these numbers to figure out what they represent.
# Discarding `"fnlwgt"` for the moment, we can identify two types of usage.
#
# The first column, `"age"`, is self-explanatory. We can note that the values
# are continuous, meaning they can take up any number in a given range. Let's
# find out what this range is:

# %%
data["age"].describe()

# %% [markdown]
# We can see the age varies between 17 and 90 years.
#
# We could extend our analysis and we will find that `"capital-gain"`,
# `"capital-loss"`, `"hours-per-week"` and `"fnlwgt"` are also representing
# quantitative data.
#
# However, the column `"education-num"` is different. It corresponds to the
# educational stage that is not necessarily the number of years studied, and
# thus not a quantitative measurement. This feature is a categorical feature
# already encoded with discrete numerical values. To see this specificity, we
# will look at the count for each educational stage:

# %%
data["education-num"].value_counts().sort_index()

# %% [markdown]
# This feature is indeed a nominal categorical feature. We exclude it from
# our analysis since particular attention is required when dealing with
# categorical features. This topic will be discussed in depth in the subsequent
# notebook.
#
# In addition, we decide to ignore the column `"fnlwgt"`. This decision is not
# linked with the feature being numerical or categorical. Indeed, this feature
# is derived from a combination other features, as mentioned in the description
# of the dataset. Thus, we will only focus on the original data collected
# during the survey.
#
# Now, we can select the subset of numerical columns and store them inside a
# new dataframe.

# %%
numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

# %% [markdown]
# ## Train-test split the dataset
#
# In the previous notebook, we loaded two separate datasets: a training one and
# a testing one. However, as mentioned earlier, having separate datasets like
# that is unusual: most of the time, we have a single one, which we will
# subdivide.
#
# We also mentioned that scikit-learn provides the helper function
# `sklearn.model_selection.train_test_split` which is used to automatically
# split the data.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42)

# %% [markdown]
# ```{tip}
# `random_state` parameter allows to get a deterministic results even if we
# use some random process (i.e. data shuffling).
# ```
#
# In the previous notebook, we used a k-nearest neighbors predictor. While this
# model is really intuitive to understand, it is not widely used.
# Here, we will make a predictive model belonging to the linear models family.
#
# ```{note}
# In short, these models find a set of weights to combine each column in the
# data matrix to predict the target. For instance, the model can come up with
# rules such as `0.1 * age + 3.3 * hours-per-week - 15.1 > 0.5` means that
# `high-income` is predicted.
# ```
#
# Thus, as we are trying to predict a qualitative property,
# we will use a logistic regression classifier.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(data_train, target_train)

# %% [markdown]
# We can now check the statistical performance of the model using the test set
# which we left out until now.

# %%
accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")

# %% [markdown]
#
# ```{caution}
# Be aware you should use cross-validation instead of `train_test_split` in
# practice. We used a single split, to highlight the scikit-learn API and the
# methods `fit`, `predict`, and `score`. In the module "Select the best model"
# we will go into details regarding cross-validation.
# ```
#
# Now the real question is: is this statitical performance relevant of a good
# predictive model? Find out by solving the next exercise!.
#
# In this notebook, we learned:
#
# * identify numerical data in a heterogeneous dataset;
# * select the subset of columns corresponding to numerical data;
# * use scikit-learn helper to separate data into train-test sets;
# * train and evaluate a more complex scikit-learn model.
