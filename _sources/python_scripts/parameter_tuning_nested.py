# %% [markdown]
# # Cross-validation and hyperparameter tuning
#
# In the previous notebooks, we saw two approaches to tune hyperparameters:
# via grid-search and randomized-search.
#
# In this notebook, we will show how to combine such hyperparameters search
# with a cross-validation.

# %% [markdown]
# ## Our predictive model
#
# Let us reload the dataset as we did previously:

# %%
from sklearn import set_config

set_config(display="diagram")

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We extract the column containing the target.

# %%
target_name = "class"
target = adult_census[target_name]
target

# %% [markdown]
# We drop from our data the target and the `"education-num"` column which
# duplicates the information from the `"education"` column.

# %%
data = adult_census.drop(columns=[target_name, "education-num"])
data.head()

# %% [markdown]
# We will create the same predictive pipeline as seen in the grid-search
# section.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))])
model

# %% [markdown]
# ## Include a hyperparameter search within a cross-validation
#
# As mentioned earlier, using a single train-test split during the grid-search
# does not give any information regarding the different sources of variations:
# variations in terms of test score or hyperparameters values.
#
# To get reliable information, the hyperparameters search need to be nested
# within a cross-validation.
#
# ```{note}
# To limit the computational cost, we affect `cv` to a low integer. In
# practice, the number of fold should be much higher.
# ```

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.05, 0.1),
    'classifier__max_leaf_nodes': (30, 40)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=2, cv=2)

cv_results = cross_validate(
    model_grid_search, data, target, cv=3, return_estimator=True)

# %% [markdown]
# Running the above cross-validation will give us an estimate of the
# testing score.

# %%
scores = cv_results["test_score"]
print(f"Accuracy score by cross-validation combined with hyperparameters "
      f"search:\n{scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# The hyperparameters on each fold are potentially different since we nested
# the grid-search in the cross-validation. Thus, checking the variation of the
# hyperparameters across folds should also be analyzed.

# %%
for fold_idx, estimator in enumerate(cv_results["estimator"]):
    print(f"Best parameter found on fold #{fold_idx + 1}")
    print(f"{estimator.best_params_}")

# %% [markdown]
# Obtaining models with unstable hyperparameters would be an issue in practice.
# Indeed, it would become difficult to set them.

# %% [markdown]
# In this notebook, we have seen how to combine hyperparameters search with
# cross-validation.
