---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Exercise 02

The goal is to find the best set of hyper-parameters which maximize the
performance on a training set.

Here again with limit the size of the training set to make computation
run faster. Feel free to increase the `train_size` value if your computer
is powerful enough.

```{code-cell}
import numpy as np
import pandas as pd

df = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = df[target_name].to_numpy()
data = df.drop(columns=[target_name, "fnlwgt"])

from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)
```

TODO: create your machine learning pipeline

You should:
* preprocess the categorical columns using a `OneHotEncoder` and use a
  `StandardScaler` to normalize the numerical data.
* use a `LogisticRegression` as a predictive model.

+++

Start by defining the columns and the preprocessing pipelines to be applied
on each columns.

```{code-cell}
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'native-country', 'sex']

categories = [data[column].unique()
              for column in data[categorical_columns]]

numerical_columns = [
    'age', 'capital-gain', 'capital-loss', 'hours-per-week']

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

categorical_processor = OneHotEncoder(categories=categories)
numerical_processor = StandardScaler()
```

Subsequently, create a `ColumnTransformer` to redirect the specific columns
a preprocessing pipeline.

```{code-cell}

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    [('cat-preprocessor', categorical_processor, categorical_columns),
     ('num-preprocessor', numerical_processor, numerical_columns)]
)
```

Finally, concatenate the preprocessing pipeline with a logistic regression.

```{code-cell}

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(preprocessor, LogisticRegression())
```

Use a `RandomizedSearchCV` to find the best set of hyper-parameters by tuning
the following parameters for the `LogisticRegression` model:
- `C` with values ranging from 0.001 to 10. You can use a reciprocal
  distribution (i.e. `scipy.stats.reciprocal`);
- `solver` with possible values being `"liblinear"` and `"lbfgs"`;
- `penalty` with possible values being `"l2"` and `"l1"`;

In addition, try several preprocessing strategies with the `OneHotEncoder`
by always (or not) dropping the first column when encoding the categorical
data.

Notes: some combinations of the hyper-parameters proposed above are invalid.
You can make the parameter search accept such failures by setting `error_score`
to `np.nan`. The warning messages give more details on which parameter
combinations but the computation will proceed.

Once the computation has completed, print the best combination of parameters
stored in the `best_params_` attribute.

```{code-cell}
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

param_distributions = {
    "logisticregression__C": reciprocal(0.001, 10),
    "logisticregression__solver": ["liblinear", "lbfgs"],
    "logisticregression__penalty": ["l2", "l1"],
    "columntransformer__cat-preprocessor__drop": [None, "first"]
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions,
    n_iter=20, error_score=np.nan, n_jobs=2, verbose=1)
model_random_search.fit(df_train, target_train)
model_random_search.best_params_
```

We could use `cv_results = model_random_search.cv_results_` in the plot at
the end of this notebook (you are more than welcome to try!). Instead we are
going to load the results obtained from a similar search with many more
iterations (200 instead of 20).

This way we can have a more detailed plot while being able to run this notebook
in a reasonably short amount of time.

```{code-cell}
# Uncomment this cell if you want to regenerate the results csv file. This
# can take a long time to execute.
#
# model_random_search = RandomizedSearchCV(
#     model, param_distributions=param_distributions,
#     n_iter=200, error_score=np.nan, n_jobs=-1)
# _ = model_random_search.fit(df_train, target_train)
# cv_results =  pd.DataFrame(model_random_search.cv_results_)
# cv_results.to_csv("../figures/randomized_search_results_logistic_regression.csv")
```

```{code-cell}
cv_results = pd.read_csv(
    "../figures/randomized_search_results_logistic_regression.csv",
    index_col=0)
```

```{code-cell}
column_results = [f"param_{name}"for name in param_distributions.keys()]
column_results += ["mean_test_score", "std_test_score", "rank_test_score"]

cv_results = cv_results[column_results].sort_values(
    "mean_test_score", ascending=False)
cv_results = (
    cv_results
    .rename(columns={
        "param_logisticregression__C": "C",
         "param_logisticregression__solver": "solver",
         "param_logisticregression__penalty": "penalty",
         "param_columntransformer__cat-preprocessor__drop": "drop",
         "mean_test_score": "mean test accuracy",
         "rank_test_score": "ranking"})
    .astype(dtype={'C': 'float64'})
)
cv_results['log C'] = np.log(cv_results['C'])
```

```{code-cell}
cv_results["drop"] = cv_results["drop"].fillna("None")
cv_results = cv_results.dropna("index").drop(columns=["solver"])
encoding = {}
for col in cv_results:
    if cv_results[col].dtype.kind == 'O':
        labels, uniques = pd.factorize(cv_results[col])
        cv_results[col] = labels
        encoding[col] = uniques
encoding
```

```{code-cell}
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.drop(columns=["ranking", "std_test_score"]),
    color="mean test accuracy",
    dimensions=["log C", "penalty", "drop",
                "mean test accuracy"],
    color_continuous_scale=px.colors.diverging.Tealrose,
)
fig.show()
```
