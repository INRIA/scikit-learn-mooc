# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📃 Solution for Exercise M3.02
#
# The goal is to find the best set of hyperparameters which maximize the
# generalization performance on a training set.
#
# Here again with limit the size of the training set to make computation
# run faster. Feel free to increase the `train_size` value if your computer
# is powerful enough.

# %%

import numpy as np
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)

# %% [markdown]
# In this exercise, we will progressively define the classification pipeline
# and later tune its hyperparameters.
#
# Our pipeline should:
# * preprocess the categorical columns using a `OneHotEncoder` and use a
#   `StandardScaler` to normalize the numerical data.
# * use a `LogisticRegression` as a predictive model.
#
# Start by defining the columns and the preprocessing pipelines to be applied
# on each group of columns.
# %%
from sklearn.compose import make_column_selector as selector

# solution
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns = numerical_columns_selector(data)

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# solution
categorical_processor = OneHotEncoder(handle_unknown="ignore")
numerical_processor = StandardScaler()

# %% [markdown]
# Subsequently, create a `ColumnTransformer` to redirect the specific columns
# a preprocessing pipeline.

# %%
from sklearn.compose import ColumnTransformer

# solution
preprocessor = ColumnTransformer(
    [('cat_preprocessor', categorical_processor, categorical_columns),
     ('num_preprocessor', numerical_processor, numerical_columns)]
)

# %% [markdown]
# Assemble the final pipeline by combining the above preprocessor
# with a logistic regression classifier. Force the maximum number of
# iterations to `10_000` to ensure that the model will converge.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# solution
model = make_pipeline(preprocessor, LogisticRegression(max_iter=10_000))

# %% [markdown]
# Use `RandomizedSearchCV` with `n_iter=20` to find the best set of
# hyperparameters by tuning the following parameters of the `model`:
#
# - the parameter `C` of the `LogisticRegression` with values ranging from
#   0.001 to 10. You can use a log-uniform distribution
#   (i.e. `scipy.stats.loguniform`);
# - the parameter `with_mean` of the `StandardScaler` with possible values
#   `True` or `False`;
# - the parameter `with_std` of the `StandardScaler` with possible values
#   `True` or `False`.
#
# Once the computation has completed, print the best combination of parameters
# stored in the `best_params_` attribute.

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# solution
param_distributions = {
    "logisticregression__C": loguniform(0.001, 10),
    "columntransformer__num_preprocessor__with_mean": [True, False],
    "columntransformer__num_preprocessor__with_std": [True, False],
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions,
    n_iter=20, error_score=np.nan, n_jobs=2, verbose=1, random_state=1)
model_random_search.fit(data_train, target_train)
model_random_search.best_params_

# %% [markdown] tags=["solution"]
#
# So the best hyperparameters give a model where the features are scaled but
# not centered and the final model is regularized.
#
# Getting the best parameter combinations is the main outcome of the
# hyper-parameter optimization procedure. However it is also interesting to
# assess the sensitivity of the best models to the choice of those parameters.
# The following code, not required to answer the quiz question shows how to
# conduct such an interactive analysis for this this pipeline using a parallel
# coordinate plot using the `plotly` library.
#
# We could use `cv_results = model_random_search.cv_results_` to make a
# parallel coordinate plot as we did in the previous notebook (you are more
# than welcome to try!). Instead we are going to load the results obtained from
# a similar search with many more iterations (1,000 instead of 20).

# %% tags=["solution"]
cv_results = pd.read_csv(
    "../figures/randomized_search_results_logistic_regression.csv")

# %% [markdown] tags=["solution"]
# To simplify the axis of the plot, we will rename the column of the dataframe
# and only select the mean test score and the value of the hyperparameters.

# %% tags=["solution"]
column_name_mapping = {
    "param_logisticregression__C": "C",
    "param_columntransformer__num_preprocessor__with_mean": "centering",
    "param_columntransformer__num_preprocessor__with_std": "scaling",
    "mean_test_score": "mean test accuracy",
}

cv_results = cv_results.rename(columns=column_name_mapping)
cv_results = cv_results[column_name_mapping.values()].sort_values(
    "mean test accuracy", ascending=False)

# %% [markdown] tags=["solution"]
# In addition, the parallel coordinate plot from `plotly` expects all data to
# be numeric. Thus, we convert the boolean indicator informing whether or not
# the data were centered or scaled into an integer, where True is mapped to 1
# and False is mapped to 0.
#
# We also take the logarithm of the `C` values to span the data on a broader
# range for a better visualization.

# %% tags=["solution"]
column_scaler = ["centering", "scaling"]
cv_results[column_scaler] = cv_results[column_scaler].astype(np.int64)
cv_results['log C'] = np.log10(cv_results['C'])

# %% tags=["solution"]
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results,
    color="mean test accuracy",
    dimensions=["log C", "centering", "scaling", "mean test accuracy"],
    color_continuous_scale=px.colors.diverging.Tealrose,
)
fig.show()

# %% [markdown] tags=["solution"]
# We recall that it is possible to select a range of results by clicking and
# holding on any axis of the parallel coordinate plot. You can then slide
# (move) the range selection and cross two selections to see the intersections.
#
# Selecting the best performing models (i.e. above an accuracy of ~0.845), we
# observe the following pattern:
#
# - scaling the data is important. All the best performing models are scaling
#   the data;
# - centering the data does not have a strong impact. Both approaches,
#   centering and not centering, can lead to good models;
# - using some regularization is fine but using too much is a problem. Recall
#   that a smaller value of C means a stronger regularization. In particular
#   no pipeline with C lower than 0.001 can be found among the best
#   models.
