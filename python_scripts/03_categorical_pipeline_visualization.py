# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Visualizing scikit-learn pipelines in Jupyter

# %% [markdown]
# The goal of keeping this notebook is to:
#
# - make it available for users that want to reproduce it locally
# - archive the script in the event we want to rerecord this video with an
#   update in the UI of scikit-learn in a future release.

# %% [markdown]
# ## First we load the dataset

# %% [markdown]
# We need to define our data and target. In this case we build a classification
# model

# %%
import pandas as pd

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values="?")

target_name = "SalePrice"
data, target = (
    ames_housing.drop(columns=target_name),
    ames_housing[target_name],
)
target = (target > 200_000).astype(int)

# %% [markdown]
# We inspect the first rows of the dataframe

# %%
data

# %% [markdown]
# For the sake of simplicity, we can cherry-pick some features and only retain
# this arbitrary subset of data:

# %%
numeric_features = ["LotArea", "FullBath", "HalfBath"]
categorical_features = ["Neighborhood", "HouseStyle"]
data = data[numeric_features + categorical_features]

# %% [markdown]
# ## Then we create the pipeline

# %% [markdown]
# The first step is to define the preprocessing steps

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# %% [markdown]
# The next step is to apply the transformations using `ColumnTransformer`

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# %% [markdown]
# Then we define the model and join the steps in order

# %%
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)
model

# %% [markdown]
# Let's fit it!

# %%
model.fit(data, target)

# %% [markdown]
# Notice that the diagram changes color once the estimator is fit.
#
# So far we used `Pipeline` and `ColumnTransformer`, which allows us to custom
# the names of the steps in the pipeline. An alternative is to use
# `make_column_transformer` and `make_pipeline`, they do not require, and do not
# permit, naming the estimators. Instead, their names are set to the lowercase
# of their types automatically.

# %%
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler()
)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features),
)
model = make_pipeline(preprocessor, LogisticRegression())
model.fit(data, target)

# %% [markdown]
# ## Finally we can score the model using cross-validation:

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
scores = cv_results["test_score"]
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)

# %% [markdown]
# ```{note}
# In this case, around 86% of the times the pipeline correctly predicts whether
# the price of a house is above or below the 200_000 dollars threshold. But be
# aware that this score was obtained by picking some features by hand, which is
# not necessarily the best thing we can do for this classification task. In this
# example we can hope that fitting a complex machine learning pipelines on a
# richer set of features can improve upon this performance level.
#
# Reducing a price estimation problem to a binary classification problem with a
# single threshold at 200_000 dollars is probably too coarse to be useful in in
# practice. Treating this problem as a regression problem is probably a better
# idea. We will see later in this MOOC how to train and evaluate the performance
# of various regression models.
# ```
