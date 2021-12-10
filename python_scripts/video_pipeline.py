# %% [markdown]
# # How to define a scikit-learn pipeline and visualize it

# %% [markdown]
# The goal of keeping this notebook is to:

# - make it available for users that want to reproduce it locally
# - archive the script in the event we want to rerecord this video with an
#   update in the UI of scikit-learn in a future release.

# %% [markdown]
# ### First we load the dataset

# %% [markdown]
# We need to define our data and target. In this case we will build a classification model

# %%
import pandas as pd

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values='?')

target_name = "SalePrice"
data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]
target = (target > 200_000).astype(int)

# %% [markdown]
# We inspect the first rows of the dataframe

# %%
data

# %% [markdown]
# We can cherry-pick some features and only retain this subset of data

# %%
numeric_features = ['LotArea', 'FullBath', 'HalfBath']
categorical_features = ['Neighborhood', 'HouseStyle']
data = data[numeric_features + categorical_features]

# %% [markdown]
# ### Then we create the pipeline

# %% [markdown]
# The first step is to define the preprocessing steps

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(),
)])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# %% [markdown]
# The next step is to apply the transformations using `ColumnTransformer`

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# %% [markdown]
# Then we define the model and join the steps in order

# %%
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()),
])

# %% [markdown]
# Let's visualize it!

# %%
from sklearn import set_config

set_config(display='diagram')
model

# %% [markdown]
# ### Finally we score the model

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
