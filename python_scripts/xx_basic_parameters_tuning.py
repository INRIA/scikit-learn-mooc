# %% [markdown]
# # Introduction to scikit-learn: basic model hyper-parameters tuning
#
# The process to learn a predictive model is driven by a set of internal
# parameters and a set of training data. These internal parameters are called
# hyper-parameters and are specific for each family of models. In addition,
# a set of parameters are optimal for a specific dataset and thus they need
# to be optimized.
#
# This notebook shows:
# * the influence of changing model parameters;
# * how to tune these hyper-parameters;
# * how to evaluate the model performance together with hyper-parameters
#   tuning.

# %%
import os
import pandas as pd

df = pd.read_csv(os.path.join('datasets', 'adult-census.csv'))
target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)
data = data.drop(columns="fnlwgt")

# %%
# preprocessing for rare category

mask_rare_category = data['native-country'] == ' Holand-Netherlands'
data = data[~mask_rare_category]
target = target[~mask_rare_category]

# %% [markdown]
# Once the dataset is loaded, we split it into a training and testing sets.

# %%
from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
# Then, we define the preprocessing pipeline to transform differently
# the numerical and categorical data.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

binary_encoding_columns = ['sex']
one_hot_encoding_columns = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race',
                            'native-country']
scaling_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'education-num']

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])

# %% [markdown]
# Finally, we use a linear classifier (i.e. logistic regression) to predict
# whether or not a person earn more than 50,000 dollars a year.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
model.fit(df_train, target_train)
print(
    f"The accuracy score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f}"
)

# %% [markdown]
# ## The issue of finding the best model parameters
#
# In the previous example, we created a `LogisticRegression` classifier using
# the default parameters by omitting setting explicitly these parameters.
#
# For this classifier, the parameter `C` governes the penalty; in other
# words, how much our model should "trust" (or fit) the training data.
# Therefore, the default value of `C` is never certified to give the best
# performing model.
#
# We can make a quick experiment by changing the value of `C` and see the
# impact of this parameter on the model performance.

# %%
C = 1
model = make_pipeline(preprocessor, LogisticRegression(C=C, max_iter=1000))
model.fit(df_train, target_train)
print(
    f"The accuracy score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with alpha={C}"
)

C = 1e-5
model = make_pipeline(preprocessor, LogisticRegression(C=C, max_iter=1000))
model.fit(df_train, target_train)
print(
    f"The accuracy score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with alpha={C}"
)

# %% [markdown]
# ## Finding the best model hyper-parameters via exhaustive parameters search
#
# We see that the parameter `C` as a significative impact on the model
# performance. This parameter should be tuned to get the best cross-validation
# score, so as to avoid over-fitting problems.
#
# In short, we will set the parameter, train our model on some data, and
# evaluate the model performance on some left out data. Ideally, we will select
# the parameter leading to the optimal performance on the testing set.
# Scikit-learn provides a `GridSearchCV` estimator which will handle the
# cross-validation and hyper-parameter search for us.

# %%
from sklearn.model_selection import GridSearchCV

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

# %% [markdown]
# We need to provide the name of the parameter to be set. We use the
# `get_params()` method to have the list of model parameters which can be
# set during the search.

# %%
print("The model hyper-parameters are:")
print(model.get_params())

# %% [markdown]
# We want to try several values for the `'logisticregression__C'` parameter.
# Let see how to use the `GridSearchCV` estimator to optimize the `C`
# parameter.

# %%
import numpy as np

param_grid = {'logisticregression__C': np.linspace(1e-5, 1, num=5)}
model_grid_search = GridSearchCV(model, param_grid=param_grid)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)

# %% [markdown]
# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all possible parameters combination. Once the grid-search fitted, it can be
# used as any other predictor by calling `predict` and `predict_proba`.
# Internally, it will use the model with the best parameters found during
# `fit`. You can know about these parameters by looking at the `best_params_`
# attribute.

# %%
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %% [markdown]
# With the `GridSearchCV` estimator, the parameters need to be specified
# explicitely. Instead, one could randomly generate (following a specific
# distribution) the parameter candidates. The `RandomSearchCV` allows for such
# stochastic search. It is used similarly to the `GridSearchCV` but the
# sampling distributions need to be specified instead of the parameter values.

# %%
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'logisticregression__C': uniform(loc=50, scale=100)}
model_grid_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=5
)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %% [markdown]
# ## Notes on search efficiency
#
# Be aware that sometimes, scikit-learn provides some `EstimatorCV` classes
# which will perform internally the cross-validation in such way that it will
# be more computationally efficient. We can give the example of the
# `LogisticRegressionCV` which can be used to find the best `alpha` in a more
# efficient way than what we previously did with the `GridSearchCV`.

# %%
import time
from sklearn.linear_model import LogisticRegressionCV

# define the different alphas to try out
param_grid = {"C": (0.1, 1.0, 10.0)}

model = make_pipeline(preprocessor, LogisticRegressionCV(Cs=param_grid['C'],
                                                         max_iter=1000))
start = time.time()
model.fit(df_train, target_train)
print(f"Time elapsed to train LogisticRegressionCV: "
      f"{time.time() - start:.3f} seconds")

model = make_pipeline(
    preprocessor, GridSearchCV(LogisticRegression(max_iter=1000),
                               param_grid=param_grid)
)
start = time.time()
model.fit(df_train, target_train)
print(f"Time elapsed to make a grid-search on LogisticRegression: "
      f"{time.time() - start:.3f} seconds")

# %% [markdown]
# ## Fine tuning a model with several parameters
#
# In the previous example, we presented how to optimized a single parameter.
# However, we can optimize several parameters. We will take an example where
# we will use the `HistGradientBoostingClassifier` in which different set of
# parameters will influence the predictive performance.

# %%
# This line is currently required to import HistGradientBoostingClassifier.
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from scipy.stats import expon
from scipy.stats import randint
from sklearn.pipeline import Pipeline

ordinal_encoding_columns = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race',
                            'native-country', 'sex']

preprocessor = ColumnTransformer(
    [('ordinal-encoder', OrdinalEncoder(), ordinal_encoding_columns)],
    remainder='passthrough', sparse_threshold=0
)

model = Pipeline(
    [('preprocessor', preprocessor),
     ('gbrt', HistGradientBoostingClassifier(max_iter=50))]
)
param_distributions = {
    'gbrt__learning_rate': expon(loc=0.001, scale=0.5),
    'gbrt__l2_regularization': uniform(loc=0, scale=0.5),
    'gbrt__max_leaf_nodes': randint(5, 30),
    'gbrt__min_samples_leaf': randint(5, 30)
}
model_grid_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=5
)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %% [markdown]
# ## Combining evaluation and hyper-parameters search
#
# Cross-validation was used for searching the best model parameters. We
# previously evaluate model performance through cross-validation as well. If we
# would like to combine both aspects, we need to perform a "nested"
# cross-validation. The "outer" cross-validation is applied to assess the
# model while the "inner" cross-validation set the hyper-parameters of the
# model on the data set provided by the "outer" cross-validation. In practice,
# it is equivalent of including, `GridSearchCV`, `RandomSearchCV`, or any
# `EstimatorCV` in a `cross_val_score` or `cross_validate` function call.

# %%
from sklearn.model_selection import cross_val_score

model = make_pipeline(preprocessor, LogisticRegressionCV(max_iter=1000))
score = cross_val_score(model, data, target)
print(f"The accuracy score is: {score.mean():.2f} +- {score.std():.2f}")
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# Be aware that such training might involve a variation of the hyper-parameters
# of the model. When analyzing such model, you should not only look at the
# overall model performance but look at the hyper-parameters variations as
# well.
