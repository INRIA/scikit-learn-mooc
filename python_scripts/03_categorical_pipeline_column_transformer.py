# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Using numerical and categorical variables together
#
# In the previous notebooks, we showed the required preprocessing to apply when
# dealing with numerical and categorical variables. However, we decoupled the
# process to treat each type individually. In this notebook, we show how to
# combine these preprocessing steps.
#
# We first load the entire adult census dataset.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])

# %% [markdown]
# ## Dispatch columns to a specific processor
#
# In the previous sections, we saw that we need to treat data differently
# depending on their nature (i.e. numerical or categorical).
#
# Skrub is a data preprocessing library built to work seamlessly with
# scikit-learn. It provides a convenient transformer called `TableVectorizer`
# that can handle both numerical and categorical variables in a single
# transformer. It makes the column selection automatically by using a column's
# `dtype`. This is equivalent to using a
# [`sklearn.compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html)
# and selecting or excluding `object` dtypes.
#
# ```{caution}
# Here, we know that `object` data type is used to represent strings and thus
# categorical features. Be aware that this is not always the case. Sometimes
# `object` data type could contain other types of information, such as dates that
# were not properly formatted (strings) and yet relate to a quantity of
# elapsed  time.
#
# In a more general scenario you should manually introspect the content of your
# dataframe not to wrongly use `make_column_selector`.
# ```
#
# `TableVectorizer` separates the columns into four groups:
# * **low cardinality categorical columns** (categorical columns with a limited
#   number of unique values, one hot encoded by default);
# * **high cardinality categorical columns** (categorical columns with a large
#   number of unique values, string encoded by default);
# * **numerical columns** (untouched by default).
# * **time columns** (columns that encode time information, as present in time
#   series for instance, converted to numerical features that can be used by
#   learners; for more information, see the
#   [documentation](https://skrub-data.org/stable/reference/generated/skrub.DatetimeEncoder.html)).
#
# The threshold to determine whether a categorical column is of low or high
# cardinality can be set using the `cardinality_threshold` parameter.

# %% [markdown]
# ## Effect of the cardinality threshold
#
# As previously stated, `TableVectorizer` separates categorical columns into two
# groups: low cardinality and high cardinality. By default, the threshold is set
# to 40 unique values. However, this value can be changed using the
# `cardinality_threshold` parameter of `TableVectorizer`. Let's vizualize its
# effect on the `"native-country"` column of the dataset. This column
# corresponds to the country of origin of each individual. Let's check how many
# unique values it contains.

# %%
data["native-country"].nunique()

#%% [markdown]
# In the setup we used so far, this column is considered as a high cardinality
# categorical column. Let us compare both encodings.

# %%
from skrub import TableVectorizer

native_country_data = data[["native-country"]]

high_thresh_vectorizer = TableVectorizer(cardinality_threshold=50)
high_card_encoded = high_thresh_vectorizer.fit_transform(native_country_data)

high_thresh_vectorizer

# %%
low_thresh_vectorizer = TableVectorizer()
low_card_encoded = low_thresh_vectorizer.fit_transform(native_country_data)

low_thresh_vectorizer

# %% [markdown] On the encoder or pipeline HTML diagrams, we can see that the
# "native-country" column has been passed as a high cardinality categorical
# column in the first case, and as a low cardinality categorical column in the
# second case by clicking the on the `low_cardinality` and `high_cardinality`
# boxes.
#
# We set the `cardinality_threshold` parameter to ensure that all the
# categorical columns are considered as low cardinality. This way, all
# categorical columns are encoded in the same manner.

# %% [markdown]
# ## Preprocessing and modeling pipeline
#
# For the rest of the notebook we apply the following transformations to the
# whole dataset:
#
# * **one-hot encoding** is applied to the low cardinality categorical columns.
#   Besides, we use `handle_unknown="ignore"` to solve the potential issues due
#   to rare categories.
# * **numerical scaling** numerical features which will be standardized.
#
# Now, we create our transformer using the helper function `TableVectorizer`. We
# specify the transformers. First, let's create the preprocessors for the
# numerical and low cardinality categorical parts.

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(
    handle_unknown="ignore", sparse_output=False
)
numerical_preprocessor = StandardScaler()

# %% [markdown]
# Now, we create the transformer and associate each of these preprocessors with
# their respective columns.

# %%
vectorizer = TableVectorizer(
    low_cardinality=categorical_preprocessor, numeric=numerical_preprocessor, cardinality_threshold=50
)

# %% [markdown]
# We can take a minute to represent graphically the structure of a
# `TableVectorizer`:
#
# ![columntransformer diagram](../figures/api_diagram-columntransformer.svg)
#
# `TableVectorizer` does the following:
#
# * It **splits the columns** of the original dataset based on the data type and
#   cardinality of unique values.
# * It **transforms each subsets**. A specific transformer is applied to each
#   subset: it internally calls `fit_transform` or `transform`. The output of
#   this step is a set of transformed datasets.
# * It then **concatenates the transformed datasets** into a single dataset.
#
# The important thing is that `TableVectorizer` is like any other scikit-learn
# transformer. In particular it can be combined with a classifier in a
# `Pipeline`:

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(vectorizer, LogisticRegression(max_iter=500))
model

# %% [markdown]
# The final model is more complex than the previous models but still follows the
# same API (the same set of methods that can be called by the user):
#
# - the `fit` method is called to preprocess the data and then train the
#   classifier of the preprocessed data;
# - the `predict` method makes predictions on new data;
# - the `score` method is used to predict on the test data and compare the
#   predictions to the expected test labels to compute the accuracy.
#
# Let's start by splitting our data into train and test sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
#
# ```{caution}
# Be aware that we use `train_test_split` here for didactic purposes, to show
# the scikit-learn API. In a real setting one might prefer to use
# cross-validation to also be able to evaluate the uncertainty of our estimation
# of the generalization performance of a model, as previously demonstrated.
# ```
#
# Now, we can train the model on the train set.

# %%
_ = model.fit(data_train, target_train)

# %% [markdown]
# Then, we can send the raw dataset straight to the pipeline. Indeed, we do not
# need to make any manual preprocessing (calling the `transform` or
# `fit_transform` methods) as it is already handled when calling the `predict`
# method. As an example, we predict on the five first samples from the test set.

# %%
data_test

# %%
model.predict(data_test)[:5]

# %%
target_test[:5]

# %% [markdown]
# To get directly the accuracy score, we need to call the `score` method. Let's
# compute the accuracy score on the entire test set.

# %%
model.score(data_test, target_test)

# %% [markdown]
# ## Effect of the cardinality threshold
#
# As previously stated, `TableVectorizer` separates categorical columns into two
# groups: low cardinality and high cardinality. By default, the threshold is set
# to 40 unique values. However, this value can be changed using the
# `cardinality_threshold` parameter of `TableVectorizer`. Let's vizualize its
# effect on the `"native-country"` column of the dataset. This column
# corresponds to the country of origin of each individual. Let's check how many
# unique values it contains.

# %%
data["native-country"].nunique()

#%% [markdown]
# In the setup we used so far, this column is considered as a high cardinality
# categorical column. Let us compare both encodings.

# %%
native_country_data = data[["native-country"]]

high_thresh_vectorizer = TableVectorizer(
    low_cardinality=OneHotEncoder(sparse_output=False), cardinality_threshold=50)
high_card_encoded = high_thresh_vectorizer.fit_transform(native_country_data)

high_thresh_vectorizer

# %%
low_thresh_vectorizer = TableVectorizer(
    low_cardinality=OneHotEncoder(sparse_output=False))
low_card_encoded = low_thresh_vectorizer.fit_transform(native_country_data)


low_thresh_vectorizer

# %% [markdown]
# On the encoder or pipeline HTML diagrams, we can see that the "native-country"
# column has been passed as a high cardinality categorical column in the first
# case, and as a low cardinality categorical column in the second case by
# clicking the on the `low_cardinality` and `high_cardinality` boxes.
#
# We set the `cardinality_threshold` parameter to ensure that all the
# categorical columns are considered as low cardinality. This way, all
# categorical columns are encoded in the same manner.

# %% [markdown]
# ## Evaluation of the model with cross-validation
#
# As previously stated, a predictive model should be evaluated by
# cross-validation. Our model is usable with the cross-validation tools of
# scikit-learn as any other predictors:

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
cv_results

# %%
scores = cv_results["test_score"]
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)

# %% [markdown]
# The compound model has a higher predictive accuracy than the two models that
# used numerical and categorical variables in isolation.

# %% [markdown]
# ## Fitting a more powerful model
#
# **Linear models** are nice because they are usually cheap to train, **small**
# to deploy, **fast** to predict and give a **good baseline**.
#
# However, it is often useful to check whether more complex models such as an
# ensemble of decision trees can lead to higher predictive performance. In this
# section we use such a model called **gradient-boosting trees** and evaluate
# its generalization performance. More precisely, the scikit-learn model we use
# is called `HistGradientBoostingClassifier`. Note that boosting models will be
# covered in more detail in a future module.
#
# For tree-based models, the handling of numerical and categorical variables is
# simpler than for linear models:
# * we do **not need to scale the numerical features**
# * using an **ordinal encoding for the categorical variables** is fine even if
#   the encoding results in an arbitrary ordering
#
# Therefore, for `HistGradientBoostingClassifier`, the preprocessing pipeline is
# slightly simpler than the one we saw earlier for the `LogisticRegression`:

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import ToCategorical

categorical_preprocessor = ToCategorical()

preprocessor = TableVectorizer(low_cardinality=categorical_preprocessor)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

# %% [markdown]
# Now that we created our model, we can check its generalization performance.

# %%time
_ = model.fit(data_train, target_train)

# %%
model.score(data_test, target_test)

# %% [markdown]
# We can observe that we get significantly higher accuracies with the Gradient
# Boosting model. This is often what we observe whenever the dataset has a large
# number of samples and limited number of informative features (e.g. less than
# 1000) with a mix of numerical and categorical variables.
#
# This explains why Gradient Boosted Machines are very popular among datascience
# practitioners who work with tabular data.

# %% [markdown]
# In this notebook we:
#
# * used a `TableVectorizer` to apply different preprocessing for categorical
#   and numerical variables;
# * used a pipeline to chain the `TableVectorizer` preprocessing and logistic
#   regression fitting;
# * saw that **gradient boosting methods** can outperform **linear models**.
