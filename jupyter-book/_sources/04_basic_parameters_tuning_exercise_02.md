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
    data, target, random_state=42)
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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
```

Subsequently, create a `ColumnTransformer` to redirect the specific columns
a preprocessing pipeline.

```{code-cell}

from sklearn.compose import ColumnTransformer
```

Finally, concatenate the preprocessing pipeline with a logistic regression.

```{code-cell}

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
```

TODO: make your random search

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
