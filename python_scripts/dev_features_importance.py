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
# In this notebook, we will detail methods to investigate the importance of features used by a given a model.
# We will look at:
#
# - interpreting coefficient in linear model;
# - the attribute `feature_importances_` in RandomForrest;
# - `permutation feature importance`, which is an inspection technique that can be used for any fitted model.

# %% [markdown]
# ## Presentation of the dataset

# %% [markdown]
# The data we will look at is a record of neighborhoods in California district, predicting the **median house value** (target) given some information about the neighborhoods, as average number of rooms, latitude, longitude and the median income

# %%
from sklearn.datasets import fetch_california_housing
import pandas as pd

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X.head()

# %% [markdown]
# The feature reads as follow:
# - MedInc median income in block
# - HouseAge median house age in block
# - AveRooms average number of rooms
# - AveBedrms average number of bedrooms
# - Population block population
# - AveOccup average house occupancy
# - Latitude house block latitude
# - Longitude house block longitude

# %%
y.head()

# %%
import numpy as np

# Adding some random variables
rng = np.random.RandomState(0)
cat_var = pd.Series(rng.randint(0,3, X.shape[0]), name = 'rnd_cat')
num_var = pd.Series(np.arange(X.shape[0]), name = 'rnd_num') # pd.Series(rng.randint(0,100, X.shape[0]), name = 'rnd_num') 
X_with_rnd_feat = pd.concat((X, cat_var, num_var), axis = 1)


# %% [markdown]
# We will split the data here for the remaining part of this notebook

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_with_rnd_feat,
                                    y, random_state = 29)

# %% [markdown]
# Let's quickly inspect some features and the target

# %%
import seaborn as sns

train_dataset = X_train.copy()
train_dataset.insert(0, "MedHouseVal", y_train)
_ = sns.pairplot(train_dataset[['MedHouseVal', 'AveBedrms', 'AveRooms', 'MedInc']], kind='reg', diag_kind='kde')

# %% [markdown]
# We see in the upper right plot that the median income seems to be positively correlated to the median house price (the target).
# Moreover, AveRooms and AveBedrms are also strongly correlated

# %% [markdown]
# ## Linear model inspection

# %% [markdown]
# In linear models, the target value is modeled as a linear combination of the features
#
# Coefficients represent the relationship between the given feature $X_i$ and the target $y$, assuming that all the other features remain constant (conditional dependence). 
#
# This is different from plotting $X_i$ versus $y$ and fitting a linear relationship: in that case all possible values of the other features are taken into account in the estimation (marginal dependence).
#
# This example will provide some hints in interpreting coefficient in linear models,

# %% [markdown]
#

# %%
from sklearn.linear_model import RidgeCV

model = RidgeCV()

model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

# %%
feature_names = X_train.columns
coefs = pd.DataFrame(
    model.coef_,
    columns=['Coefficients'], index=feature_names
)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('Ridge model')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# %% [markdown]
# The AveBedrms have the higher coefficient. However, we can't compare the magnitude of the coefficient directly, since there are not scaled.
# Indeed, population is a integer which can be thousand, while AvergaeBedrooms is around 4 and Latitude are in degree.
#
# The Population coefficient is expressed in "dollar / habitant" while the AveBedrms is expressed in "dollar / nb of bedrooms" and the Latitude coefficient on "dollar / degree". 
#
# We see that changing population by one do not change the outcome, while as we go south (lattitude increase) the price become cheaper.
#
#

# %% [markdown]
#

# %% [markdown]
# Looking at the coefficient plot to gauge feature importance can be misleading as some of them vary on a small scale, while others, varies a lot more, several decades.
#
# This become visible if we compare the standard deviations of different features.

# %%
X_train.std(axis=0).plot(kind='barh', figsize=(9, 7))
plt.title('Features std. dev.')
plt.subplots_adjust(left=.3)
plt.xlim((0,100))

# %% [markdown]
# So we scale everything, removing the mean, variance goes to 1.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(),
                      RidgeCV(),
                      )

model.fit(X_train, y_train)

# %%
feature_names = X_train.columns
coefs = pd.DataFrame(
    model[1].coef_,
    columns=['Coefficients'], index=feature_names
)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('Ridge model, small regularization')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# %% [markdown]
# Averagebedrooms and Houseage are rescaled
# The mdeian income feature now become the most important feature for our model, which is more aligned with our intuition.
#
#
# The plot above tells us about dependencies between a specific feature and the target when all other features remain constant, i.e., conditional dependencies. An increase of the House age will induce a increase of the price when all other features remain constant. On the contrary, an increase of the average rooms will induce an decrease of the price when all other features remain constant. Also, median income, latitude and longitude are the three variables that most influence the model.

# %% [markdown]
# ## Checking the variability of the coefficients

# %% [markdown]
# We can check the coefficient variability through cross-validation: it is a form of data perturbation.
#
# If coefficients vary significantly when changing the input dataset their robustness is not guaranteed, and they should probably be interpreted with caution.

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

cv_model = cross_validate(
    model, X_with_rnd_feat, y, cv=RepeatedKFold(n_splits=5, n_repeats=5),
    return_estimator=True, n_jobs=-1
)
coefs = pd.DataFrame(
    [model[1].coef_
     for model in cv_model['estimator']],
    columns=feature_names
)
plt.figure(figsize=(9, 7))
sns.swarmplot(data=coefs, orient='h', color='k', alpha=0.5)
# sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5)
plt.axvline(x=0, color='.5')
plt.xlabel('Coefficient importance')
plt.title('Coefficient importance and its variability')
plt.subplots_adjust(left=.3)

# %% [markdown]
# Every coefficient looks pretty stable, different model put almost the same weight to the same feature.

# %% [markdown]
# ## Lasso

# %%
from sklearn.linear_model import Lasso

model = make_pipeline(StandardScaler(),
                      Lasso(alpha=.03))

model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

# %%
feature_names = X_train.columns
coefs = pd.DataFrame(
    model[1].coef_,
    columns=['Coefficients'], index=feature_names
)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('Ridge model, small regularization')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# %% [markdown]
# ### Lessons learned
#
# Coefficients must be scaled to the same unit of measure to retrieve feature importance. Scaling them with the standard-deviation of the feature is a useful proxy.
#
# Coefficients in multivariate linear models represent the dependency between a given feature and the target, conditional on the other features.
#
# Correlated features induce instabilities in the coefficients of linear models and their effects cannot be well teased apart.
#
# Different linear models respond differently to feature correlation and coefficients could significantly vary from one another.
#
# Inspecting coefficients across the folds of a cross-validation loop gives an idea of their stability.

# %%

# %% [markdown]
# ## RandomForest feature_importances_
#
# On some algorithm, there pre-exist some feature importance method, inherently build within the model. 
# It is the case for RandomForrest models. Let's compare the build-in attribute on our model.

# %%
model = RandomForestRegressor()

model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

# %%
importances = model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])

ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(X_train.columns)[indices])

# %% [markdown]
# xx
# It also has a small bias toward high cardinality feature, such as `rnd_num`, whic are here predicted having .07 importance.
#

# %% [markdown]
# ## Feature importance by permutation
#
# The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled

# %% [markdown]
# We are now able to fit our model. Since the feature importance algotirgm that we present is model agnostic, it will not depend on the model we choose here. 
# The objectif is however to have a model powerful enough, so intepreting its feature importance shall be relevant.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import make_pipeline

clf = RandomForestRegressor()
clf = RidgeCV()
clf = KNeighborsRegressor()
model = make_pipeline(StandardScaler(),
                      clf)

# %%
clf.fit(X_train, y_train)

# %%
model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')


# %% [markdown]
# The score on test set is .81, so feature importance shall be relevant here for this model. 
# Note that on train set, the score is perfect, it means that the model do overfit completly the training data.

# %% [markdown]
# ### Feature importance 

# %% [markdown]
# Lets compute the feature importance for a given feature, say the `MedInc` feature.
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

curr_feat = 'MedInc'

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
# 0.83 over .79 is very relevant (notne the R2 score could go below 0). So we can imagine our model really evely on this feature to predict the class.
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

# This function could directly be access from sklearn
# from sklearn.inspection import permutation_importance


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

# %%
perm_importance_result_train = permutation_importance(model, X_train, y_train,
                           n_repeats=10)

# %%
import matplotlib.pyplot as plt

plot_importantes_features(perm_importance_result_train, X_train.columns)

# %% [markdown]
# We see again that the feature `Latitude`, `Longitude` and `MedInc` are very important for the model.
#
# We note that our random variable `rnd_num` xx
# that our model do use these feature to compute the output. It is in line with the overfitting we had notice between the train and test score.
#
# Now we compute the feature importance by permutation on the *testing data*.

# %%
perm_importance_result_test = permutation_importance(model, X_test, y_test,
                           n_repeats=10)

plot_importantes_features(perm_importance_result_test, X_test.columns)

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
