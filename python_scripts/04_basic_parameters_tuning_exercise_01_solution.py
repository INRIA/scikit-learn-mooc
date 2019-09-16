# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# #  Exercise 01
# The goal is to find the best set of hyper-parameters which maximize the
# performance on a training set.

# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# This line is currently required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from scipy.stats import expon, uniform
from scipy.stats import randint

df = pd.read_csv("https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
# You should:
# - create a preprocessor using an `OrdinalEncoder`
# - use a `HistGradientBoostingClassifier` to make predictions
# - use a `RandomizedSearchCV` to find the best set of hyper-parameters by
#   tuning the following parameters: `learning_rate`, `l2_regularization`,
#   `max_leaf_nodes`, and `min_samples_leaf`.

# %%
ordinal_encoding_columns = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race',
                            'native-country', 'sex']

categories = [data[column].unique()
              for column in data[ordinal_encoding_columns]]

preprocessor = ColumnTransformer(
    [('ordinal-encoder', OrdinalEncoder(categories=categories),
      ordinal_encoding_columns)],
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
    model, param_distributions=param_distributions, n_iter=10, n_jobs=4
)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %%
df_results = pd.DataFrame(model_grid_search.cv_results_)
columns = (['mean_test_score', 'std_test_score'] +
           [col for col in df_results.columns if 'param_' in col])
df_results.sort_values(by='mean_test_score', ascending=False)[
    columns
]

# %%
