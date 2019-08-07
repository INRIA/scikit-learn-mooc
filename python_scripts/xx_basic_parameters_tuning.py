# %%
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import uniform

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# %%
# Loading data
df = pd.read_csv(os.path.join('datasets', 'cps_85_wages.csv'))
target_name = "WAGE"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

# %%
# split the data

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %%
# Define the preprocessing

binary_encoding_columns = ['MARR', 'SEX', 'SOUTH', 'UNION']
one_hot_encoding_columns = ['OCCUPATION', 'SECTOR', 'RACE']
scaling_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])

# %%
model = make_pipeline(preprocessor, Ridge(alpha=1))
start = time.time()
model.fit(df_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds"
)

# %%
model = make_pipeline(preprocessor, Ridge(alpha=10000))
start = time.time()
model.fit(df_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds"
)

# %%
# we can use grid-search
model = make_pipeline(preprocessor, Ridge())
param_grid = {'ridge__alpha': np.linspace(0.001, 1000, num=20)}
model_grid_search = GridSearchCV(model, param_grid=param_grid)
start = time.time()
model_grid_search.fit(df_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f} with a fitting time "
    f"of {elapsed_time:.3f} seconds"
)
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %%
# similarly we could use a random-search
model = make_pipeline(preprocessor, Ridge())
param_distributions = {'ridge__alpha': uniform(loc=50, scale=100)}
model_grid_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=20
)
start = time.time()
model_grid_search.fit(df_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f} with a fitting time "
    f"of {elapsed_time:.3f} seconds"
)
print(f"The best set of parameters is: {model_grid_search.best_params_}")

# %%
# Some predictors come with internal cross-validation to fix hyperparameter
# and they are sometimes more efficient than using a grid-search

model = make_pipeline(preprocessor, RidgeCV())
start = time.time()
model.fit(df_train, target_train)
print(f"Time elapsed: {time.time() - start} sec")

param_grid = {"alpha": (0.1, 1.0, 10.0)}
model = make_pipeline(
    preprocessor, GridSearchCV(Ridge(), param_grid=param_grid)
)
start = time.time()
model.fit(df_train, target_train)
print(f"Time elapsed: {time.time() - start} sec")

# %%
# hyper parameters search in nested cross-validation
model = make_pipeline(preprocessor, RidgeCV())
start = time.time()
score = cross_val_score(model, data, target)
print(f"Time elapsed: {time.time() - start} sec")
print(score)

# %%
