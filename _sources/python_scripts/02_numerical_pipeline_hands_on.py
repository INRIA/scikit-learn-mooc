# %% [markdown]
# # Working with numerical data
#
# In the previous notebook, we trained a k-nearest neighbors on some data.
# However, we oversimplify the procedure by loading a dataset that only
# contained numerical data. Besides, we used datasets which were already
# split into train-test sets.
#
# In this notebook, we aim at:
#
# * identifying numerical data in a heterogeneous dataset;
# * select the subset of columns corresponding to numerical data;
# * use scikit-learn helper to separate data into train-test sets;
# * train and evaluate a more complex scikit-learn model.
#
# We will start by loading the adult census dataset used during the data
# exploration.
#
# ## Loading the entire dataset
#
# As in the previous notebook, we rely on Pandas to open the CSV file into
# a dataframe.

# %%
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")
df.head()

# %% [markdown]
# The next step is to separate the target from the data. We will replacate the
# same step than in the previous notebook.

# %%
data, target = df.drop(columns="class"), df["class"]

# %%
data.head()

# %%
target

# %% [markdown]
# At this stage, we can focus on the data that we want to use to train or
# future predictive model.
#
# ## Identify numerical data
#
# Numerical data are represented with numbers. They are linked to data that is
# measurable such as age or the number of hours a person works per week.
#
# Predictive models are designed to work with numerical data natively and it
# is a type of data that require a small amount of work to get started.
#
# The first task here will be to identify numerical data in our dataset. As we
# mentioned, numerical data are represented with numbers, but numbers are not
# always representing numerical data. Thus, we can check the data type for each
# of the column in the dataset.

# %%
data.dtypes

# %% [markdown]
# We observe two types data type. We can make sure by checking the unique data
# types.

# %%
data.dtypes.unique()

# %% [markdown]
# Thus, we see two types of data types: integer and object. We can look at
# the first few lines of the dataframe to understand the meaning of the
# `object` data types.

# %%
data.head()

# %% [markdown]
# We see that `object` data type corresponds to columns containing strings. As
# we saw in the exploration section, these columns contains categories and we
# will see later how to handle those. We can select the columns containing
# integers and check their content.

# %%
numerical_columns = [
    "age", "education-num", "capital-gain", "capital-loss",
    "hours-per-week", "fnlwgt"]
data[numerical_columns].head()

# %% [markdown]
# Now that we limited the dataset to numerical columns only, we can check
# closely what these numbers represent. Discarding `"fnlwgt"` aside for the
# moment, we can identify two types of usage.
#
# The former is related to a measurement such as age. The data are continuous
# meaning that they can take any value in a range. We can give the range for
# age column as an example,

# %%
data["age"].describe()

# %% [markdown]
# The age varies between 17 years and 90 years and can take any value in this
# range.
#
# The latter is related categorical data. These data are discrete, in contrast
# with continuous. It means that the variable can take only certain values
# which are known as categories. We will come back later on this type of data
# and how to handle them. Here, we are only interested to recognize them.
#
# Here, the column `"education-num"` gives an example. The number encode the
# education level which can only correspond to specific values. We can quickly
# check the number of occurrence of each category to get convinced.

# %%
data["education-num"].value_counts().sort_index()

# %% [markdown]
# Therefore, we should ignore such type of columns because they would require
# a specific processing which is different from the continuous variable.
#
# Finally, we can mention that we will ignore the `"fnwgt"` column because it
# corresponds to an hand-crafted variable and we make the choice to only work
# with variable which has been collected. In the next notebook, we will
# regularly ignore this variable as well.
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
# In the previous notebook, we loaded separately two datasets: a training and a
# testing dataset. We mentioned that scikit-learn provides an helper function
# `sklearn.model_selection.train_test_split` allowing to do this split.
# Here, we will use this tool instead of loading some new data.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42)

# %% [markdown]
# We recall that the `random_state` parameter allows to get a deterministic
# results even if we use some random process (i.e. data shuffling).
#
# In the previous notebook, we used a k-nearest neighbors predictor. While this
# model is really intuitive to understand, it is not widely used. Here, we will
# a predictive model belonging to the linear model families. In short, these
# models find a set of weights to combine each column in the data matrix to
# predict the target. Thus, we will use a logistic regression classifier and
# train it.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(data_train, target_train)

# %% [markdown]
# We can now check the performance of the model using the test set which we
# left out until now.

# %%
accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regresssion: {accuracy:.3f}")

# %% [markdown]
# Now the real question is: is this performance relevant of a good predictive
# model? You will answer to this question by solving the next exercise.
#
# In this notebook, we have learnt to:
#
# * identify numerical data in a heterogeneous dataset;
# * select the subset of columns corresponding to numerical data;
# * use scikit-learn helper to separate data into train-test sets;
# * train and evaluate a more complex scikit-learn model.
