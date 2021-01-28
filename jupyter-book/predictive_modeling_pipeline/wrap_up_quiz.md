# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `house_prices.csv` with the following command:

```py
import pandas as pd
data = pd.read_csv("../datasets/house_prices.csv", na_values="?")
data = data.drop(columns="Id")

target_name = "SalePrice"
X, y = data.drop(columns=target_name), data[target_name]
y = (y > 200_000).astype(int)
```

`data` is a pandas dataframe. The column "SalePrice" contains the target
variable. Note that we instructed pandas to treat the character "?" as a marker for cells with missing values also known as "null" values.

Furthermore, we ignore the column named "Id" because unique identifiers are
usually useless in the context of predictive modeling.

We did not encounter any regression problem yet. Therefore, we will convert the
regression target into a classification target to predict whether or not an
house is expensive. "Expensive" is defined as a sale price greater than
$200,000.

```{admonition} Question
Use the `data.info()` command to examine the columns of the dataframe. The dataset contains:
_Select several answers_

- a) numerical features
- b) categorical features
- c) missing data
```

+++

```{admonition} Question
How many features are available to predict whether or not an house is
expensive?
_Select a single answer_

- a) 79
- b) 80
- c) 81
```

+++

```{admonition} Question
How many features are represented with numbers?
_Select a single answer_

- a) 0
- b) 36
- c) 42
- d) 79

Hint: you can use the method
[`df.select_dtypes`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)
or as shown in the lecture, the function
[`sklearn.compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html)
```

+++

```{admonition} Question
Among the following columns, which columns express a quantitative numerical value (excluding ordinal categories)? Refer to the
[dataset description](https://www.openml.org/d/42165) regarding the meaning of the dataset.
_Select several answers_

- a) "LotFrontage"
- b) "LotArea"
- c) "OverallQual"
- d) "OverallCond"
- e) "YearBuilt"

```

+++

We consider the following numerical columns:

```py
numerical_features = [
"LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
"BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
"GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
"GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
"3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
```

```{admonition} Question
Create a predictive model that will use these numerical columns as input data.
Your predictive model should be a pipeline composed of a scaler, a mean imputer
(cf. [`sklearn.impute.SimpleImputer(strategy="mean")`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html))
and a logistic regression classifier.

What is the accuracy score obtained by cross-validation of your predictive
model?
_Select a single answer_

- a) ~0.5
- b) ~0.7
- c) ~0.9
```

+++

```{admonition} Question
Instead of solely using the numerical columns, encode the left-out columns
using a one-hot encoder. Before to one-hot encode, impute the missing values
with an imputer that will replace missing values by the most-frequent value in
the column (cf. `strategy="most_frequent` in the
`sklearn.impute.SimpleImputer`).

With this heterogeneous pipeline, the accuracy score:
_Select a single answer_

- a) worsen substantially
- b) worsen slightly
- c) improve slightly
- d) improve substantially

Hint: a substantial improvement or deterioration is respectively defined as an
increase or decrease of the mean generalization score of at least three times
the standard deviation of the generalization score.
```
