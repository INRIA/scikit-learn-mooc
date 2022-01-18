# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `ames_housing_no_missing.csv` with the following command:

```python
import pandas as pd
ames_housing = pd.read_csv("../datasets/ames_housing_no_missing.csv")

target_name = "SalePrice"
data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]
target = (target > 200_000).astype(int)
```

`ames_housing` is a pandas dataframe. The column "SalePrice" contains the
target variable.

We did not encounter any regression problem yet. Therefore, we convert the
regression target into a classification target to predict whether or not an
house is expensive. "Expensive" is defined as a sale price greater than
$200,000.

```{admonition} Question
Use the `data.info()` and ` data.head()` commands to examine the columns of
the dataframe. The dataset contains:

- a) only numerical features
- b) only categorical features
- c) both numerical and categorical features

_Select several answers_
```

+++

```{admonition} Question
How many features are available to predict whether or not a house is
expensive?

- a) 79
- b) 80
- c) 81

_Select a single answer_
```

+++

```{admonition} Question
How many features are represented with numbers?

- a) 0
- b) 36
- c) 42
- d) 79

_Select a single answer_

Hint: you can use the method
[`df.select_dtypes`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)
or the function
[`sklearn.compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html)
as shown in a previous notebook.
```

+++

Refer to the [dataset description](https://www.openml.org/d/42165) regarding
the meaning of the dataset.

```{admonition} Question
Among the following columns, which columns express a quantitative numerical
value (excluding ordinal categories)?

- a) "LotFrontage"
- b) "LotArea"
- c) "OverallQual"
- d) "OverallCond"
- e) "YearBuilt"

_Select several answers_
```

+++

We consider the following numerical columns:

```python
numerical_features = [
  "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
  "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
  "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
  "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
```

Now create a predictive model that uses these numerical columns as input data.
Your predictive model should be a pipeline composed of a
[`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
to scale these numerical data and a
[`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

```{admonition} Question
What is the accuracy score obtained by 5-fold cross-validation of this
pipeline?

- a) ~0.5
- b) ~0.7
- c) ~0.9

_Select a single answer_
```

+++

Instead of solely using the numerical columns, let us build a pipeline that
can process both the numerical and categorical features together as follows:

- numerical features should be processed as previously done with a
  `StandardScaler`;
- the left-out columns should be treated as categorical variables using a
  [`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
  To avoid any issue with rare categories that could only be present during
  the prediction, you can pass the parameter `handle_unknown="ignore"` to the
  `OneHotEncoder`.

Let us now define a **substantial** improvement or deterioration as an
increase or decrease of the mean test score (**difference of the mean
test scores** of models using only numerical features and numerical
together with categorical features) of **at least three times the
standard deviation** of the cross-validated test scores of the model
using both categorical and numerical features.

```{admonition} Question
With this heterogeneous pipeline, the accuracy score:

- a) worsens substantially
- b) worsens slightly
- c) improves slightly
- d) improves substantially

_Select a single answer_
```
