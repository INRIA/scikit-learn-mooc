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
# # Feature importance
#
# In this notebook, we will go into details on an algorithm to evaluating the importance of features for a given a model.
#
# Permutation feature importance is a model inspection technique that can be used for any fitted estimator when the data is tabular. 

# %% [markdown]
# ## Presentation of the titanic dataset

# %% [markdown]
# We use the titanic dataset, which is composed of the record of passagenr of the titanic 
# the class to predict is wether the passagenr had survived from the titanic 
#

# %%
# Load Titnaic
import pandas as pd
from sklearn.datasets import fetch_openml

# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
titanic = pd.concat((X, y), axis=1)

# %% [markdown]
# In order to assess the feature importance, we will add some random feature onto our dataset. We append one "categorical" liike variable, and one variable with a lot of different numerical value

# %%
import numpy as np

# Adding some random variables
cat_var = pd.Series(np.random.randint(0,3, X.shape[0]), name = 'rnd_cat')
num_var = pd.Series(np.random.randint(0,50000, X.shape[0]), name = 'rnd_num')
X = pd.concat((X, cat_var, num_var), axis = 1)



# %%
X.head()
# SibSp: Number of siblings/spouses aboard
# Parch: Number of parents/children aboard

# %%
y.head()

# %%
# explore the data 
import matplotlib.pyplot as plt

columns = ['sex', 'age', 'pclass', 'sibsp']

for col in columns :
    plt.figure()
    plt.hist(X.iloc[np.where(y==y[0])][col], alpha = .5)
    plt.hist(X.iloc[np.where(y==y[2])][col], alpha = .5)
    

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# %% [markdown]
# ### fit the model
# For simplicity, we will focus on numericla and categorical feature.
# Since teh data contains some missing values, we will impute them: the numerical missing value will be impute by the `median`, the categorical missing value will be replace by the categorie "missing".
# We then one hot encode the categorical features.

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ['age', 'fare', 'sibsp', 'rnd_num']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass', 'rnd_cat']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# %% [markdown]
# We are able to fit our model. Since the feature importance algotirgm that we present is model agnostic, it will not depend on the model we choose here. 
# The objectif is to have a model powerful enough, so intepreting it shall be relevant.

# %%

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

clf = RandomForestClassifier()
# clf = LogisticRegression(solver='lbfgs', max_iter=1000)
# clf = KNeighborsClassifier()
model = make_pipeline(preprocessor,
                      clf)

# %%
_ = model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')


# %% [markdown]
# The score on test set is .79, so feature importance shall be relevant here for this model. 
# Note that on train set, the score is almost perfect, it means that the model do overfit completly the training data.

# %% [markdown]
# ### Feature importance 

# %% [markdown]
# The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled

# %% [markdown]
# Lets compute the feature importance for a given feature. say it will be `pclass` (i.e. the passagenr class on the titanic)
#
# For that, we will shuffle this specific feature, keeping the other feature as is, and run our same model to predict the outcome.
# The decrease of the score shall indicate how the model had used this feature to predict the target.
#
# For instance, if the feature is crucial for the model, the outcome would also be permuted (just as the feature), thus the score would be close to zero. Afterward, the feature importance is compute as being egal to the decrease in score. So in that case, the feature importance would be close to the score.
#
# On the contrary, if the feature is not used by the model, the score shall remain the same, thus the feature importance will be close to 0.

# %%
def get_score_after_permutation(model, X, y, curr_feat):
    """ return the score of model when curr_feat is permuted """
    
    # permute one column in pandas DataFrame
    X_permuted = X.copy()
    col_idx = list(X.columns).index(curr_feat)
    X_permuted.iloc[:, col_idx] = np.random.permutation(X_permuted[curr_feat].values)
    
    permuted_score = model.score(X_permuted, y)
    return permuted_score


def get_feature_importance(model, X, y, curr_feat):
    """ compare the score when curr_feat is permuted """
    baseline_score_train = model.score(X, y)
    permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)
    
    # feature importance is the difference between the two scores 
    feature_importance = baseline_score_train - permuted_score_train
    return feature_importance

curr_feat = 'hours-per-week'
curr_feat = 'age'
curr_feat = 'pclass'

feature_importance = get_feature_importance(model, X_train, y_train, curr_feat)
print(f'feature importance of "{curr_feat}" on train set is {feature_importance:.3}')

# %% [markdown]
# Since there are some randomness, it is advice to run multiple times and inscpect the mean and the standard deviation of the feature importance

# %%
n_repeats = 10

list_feature_importance = []
for n_round in range(n_repeats):
    list_feature_importance.append(get_feature_importance(model, X_train, y_train, curr_feat))
    
print(f'feature importance of "{curr_feat}" on train set is '
      f'{np.mean(list_feature_importance):.3} +/- {np.std(list_feature_importance):.3}')


# %% [markdown]
# We can now compute the feature permutation importance for all the features

# %%
def permutation_importance(model, X, y, n_repeats = 10):
    """Calculate importance score for each feature."""
    
    importances = []
    for curr_feat in X.columns:
        list_feature_importance = []
        for n_round in range(n_repeats):
            list_feature_importance.append(get_feature_importance(model, X, y, curr_feat))

        importances.append(list_feature_importance)

    return {'importances_mean': np.mean(importances, axis=1),
                'importances_std': np.std(importances, axis=1),
                'importances': importances}



# %% [markdown]
# This function could be access by `from sklearn.inspection import permutation_importance`

# %%
def plot_importantes_features(perm_importance_result, feat_name):
    """ bar plot the feature importance """
    
    fig, ax = plt.subplots()
    
    indices = perm_importance_result['importances_mean'].argsort()
    plt.barh(range(len(indices)),
             perm_importance_result['importances_mean'][indices],
             xerr=perm_importance_result['importances_std'][indices])

    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])


# %% [markdown]
# Let's compute the feature importance by permutation on the training data.
# What the model learnt 

# %%
perm_importance_result_train = permutation_importance(model, X_train, y_train,
                           n_repeats=10)

plot_importantes_features(perm_importance_result_train, X_train.columns)

# %% [markdown]
# We see that the feature `sex` and `pclass` are very important for the model, since it represent a decrease of .30 and .18 in a score of .99.
#
# We note that our random variable `rnd_cat` and `rnd_num` has both .10 of importance score. it means that our model do use these feature to compute the output. It is in line with the overfitting we had notice between the train and test score.
#

# %% [markdown]
# Now we compute the feature importance by permutation on the *testing data*.

# %%
perm_importance_result_test = permutation_importance(model, X_test, y_test,
                           n_repeats=30)

plot_importantes_features(perm_importance_result_test, X_test.columns)

# %% [markdown]
# We note `sex`, `pclass` and `age` are used by the model.
# On the contrary the two random variables are now without importance on the test score.

# %%

# %% [markdown]
# # Random Forest feature importances_
#
# On some algorithm, there pre-exist some feature importance method, inherently build within the model. 
# It is the case for RandomForrest models. Let's compare the build-in attribute on our model.

# %%
rf = RandomForestClassifier()
model = make_pipeline(preprocessor,
                      rf)

_ = model.fit(X_train, y_train)

# %% [markdown]
#

# %%
feat_name = list(model[0].transformers_[1][1][1].get_feature_names()) + list(numeric_features)

# %%
importances = model[1].feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])

ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(feat_name)[indices][::-1])

# %% [markdown]
# The feature_importances_ also predict that the `pclass` and the `sex` is iportant for the model. 
# It also has a small bias toward high cardinality feature, such as `rnd_num`.
#

# %% [markdown]
# # Warnings / discussion
#
# 1. Dropping the column will create another model. So you would not analyse the perf of YOUR model.
#
# 2. Correlated feature: permutation could give non realistic sample (e.g. height and weight of a person)
#
# 3. It is unclear whether you should use training or testing data to compute the feature importance.

# %%

# %% [markdown]
# # Take Away
#
#

# %%
