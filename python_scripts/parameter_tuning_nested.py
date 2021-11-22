# %% [markdown]
# # Evaluation and hyperparameter tuning
#
# In the previous notebook, we saw two approaches to tune hyperparameters.
# However, we did not present a proper framework to evaluate the tuned models.
# Instead, we focused on the mechanism used to find the best set of parameters.
#
# In this notebook, we will reuse some knowledge presented in the module
# "Selecting the best model" to show how to evaluate models where
# hyperparameters need to be tuned.
#
# Thus, we will first load the dataset and create the predictive model that
# we want to optimize and later on evaluate.
#
# ## Loading the dataset
#
# As in the previous notebook, we will load the Adult census dataset. The
# loaded dataframe will be split to get the data and the target into two
# separated variables. In addition, we will drop the column `"education-num"`
# as previously done.

# %%
import pandas as pd

target_name = "class"
adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

# %% [markdown]
# ## Our predictive model
#
# We now create the predictive model that we want to optimize. Note that
# this pipeline is identical to the one we used in the previous notebook.

# %%
from sklearn import set_config

# To get a diagram visualization of the pipeline
set_config(display="diagram")

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
# ## Evaluation
#
# ### Without hyperparameter tuning
#
# In the module "Selecting the best model", we saw that one must use
# cross-validation to evaluate such a model. Cross-validation allows to get
# a distribution of the scores of the model. Thus, having this distribution at
# hand, we can get the assess the variability of  our estimate of the generalization
# performance of the model. Here, we recall the necessary `scikit-learn` tools
# needed to obtain the mean and standard deviation of the scores.

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
# The cross-validation scores are coming from a 5-fold cross-validation. So
# we can compute the mean and standard deviation of the generalization score.

# %%
print(
    "Generalization score without hyperparameters tuning:\n"
    f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}"
)

# %% [markdown]
# Now, we will present how to evaluate the model with hyperparameter tuning,
# where an extra step is required to select the best set of parameters.
#
# ### With hyperparameter tuning
#
# As we shown in the previous notebook, one can use a search strategy that uses
# cross-validation to find the best set of parameters. Here, we will use a
# grid-search strategy and reproduce the steps done in the previous notebook.
#
# First, we have to embed our model into a grid-search and specify the
# parameters and the parameter values that we want to explore.

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.05, 0.5),
    'classifier__max_leaf_nodes': (10, 30)}
model_grid_search = GridSearchCV(
    model, param_grid=param_grid, n_jobs=2, cv=2
)
model_grid_search.fit(data, target)

# %% [markdown]
# As previously saw, when calling the `fit` method, the model embedded in the
# grid-search will be trained with every possible combinations of parameters
# resulting from the parameter grid. The best combination was selected by
# keeping the combination of the best mean accuracy score.

# %%
cv_results = pd.DataFrame(model_grid_search.cv_results_)
cv_results[[
    "param_classifier__learning_rate",
    "param_classifier__max_leaf_nodes",
    "mean_test_score",
    "rank_test_score"
]]

# %%
model_grid_search.best_params_

# %% [markdown]
# One important caveat here concerns the evaluation of the generalization performance.
# Indeed, the mean and standard deviation of the scores computed
# by the cross-validation in the grid-search is potentially not a good estimate
# of the generalization performance of the model refitted with the best
# parameters (i.e. the model at hand when using `model_grid_search.predict`).
#
# The reason is that the model was refitted on the full dataset given during
# `fit`. Therefore, this refitted model was trained with more data than the
# different models trained during the cross-validation in the grid-search.
#
# Therefore, one must keep an external, held-out test set for the final evaluation the refitted model. We
# highlight here the process using a single train-test split.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

model_grid_search.fit(data_train, target_train)
accuracy = model_grid_search.score(data_test, target_test)
print(f"Accuracy on test set: {accuracy:.3f}")

# %% [markdown]
# In the code above, the selection of the best hyperparameters was done only on
# the train set. Then, we evaluated the generalization performance of our
# tuned model on the left out test set.
#
# However, this evaluation only provides us a single point estimate of the
# generalization performance. As recall at the beginning of this notebook,
# it is beneficial to have a rough idea of the uncertainty of our estimate of the generalization
# performance. Therefore, we should instead use a cross-validation for this
# evaluation.
#
# This pattern is called *nested cross-validation*. We use an inner
# cross-validation for the selection of the hyperparameters and an outer
# cross-validation for the evaluation of generalization performance of the
# refitted tuned model.
#
# In practice, we only need to embed the grid-search in the function
# `cross-validate` to perform such evaluation.

# %%
cv_results = cross_validate(
    model_grid_search, data, target, cv=5, n_jobs=2, return_estimator=True
)

# %%
cv_results = pd.DataFrame(cv_results)
print(
    "Generalization score with hyperparameters tuning:\n"
    f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}"
)

# %% [markdown]
# In this case, we obtain a distribution of scores and therefore, we can
# apprehend the variability of our estimate of the generalization performance.
#
# In addition, passing the parameter `return_estimator=True`, we can check the
# value of the best hyperparameters obtained for each fold of the outer
# cross-validation.

# %%
for cv_fold, estimator_in_fold in enumerate(cv_results["estimator"]):
    print(
        f"Best hyperparameters for fold #{cv_fold + 1}:\n"
        f"{estimator_in_fold.best_params_}"
    )

# %% [markdown]
# It is interesting to see whether the hyper-parameter tuning procedure always select
# similar values for the hyper-parameters. If its the case, then all is fine. It means that
# we can deploy a model fit with those hyper-parameters and expect that it will have
# an actual predictive performance close to what we measured in the outer
# cross-validation.
#
# But it is also possible that some hyper-parameters do not matter at all, and as a
# result in different tuning sessions give different results. In this case,
# any value will do. This can typically be confirmed by doing a parallel coordinate plot
# of the results of a large hyper-parameter search as seen in the exercises.
#
# From a deployment, one could also chose to deploy all the models found by the outer
# cross-validation loop and make them vote to get the final predictions. However this
# can cause operational problems because it uses more memory and makes
# computing prediction slower resulting in a higher computational resource usage
# per prediction. 
#
# In this notebook, we have seen how to combine hyperparameters search with
# cross-validation.
