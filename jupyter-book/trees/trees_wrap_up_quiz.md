# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `house_prices.csv` with the following command:

```py
ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values="?")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]
```

`ames_housing` is a pandas dataframe. The column "SalePrice" contains the
target variable. Note that we instructed pandas to treat the character "?" as a
marker for cells with missing values also known as "null" values.

To simplify this exercise, we will only used the numerical features defined
below:

```
numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]
```

Compare the statistical performance of a decision tree and a linear regression.
Create two predictive models and evaluate them by 10-Fold cross-validation.

Note that missing values should be handle with a scikit-learn `SimpleImputer`
and the default strategy. Be also aware that the linear model requires to scale
the data. You can use a `StandardScaler`.

Use the default parameter for both `LinearRegression` and
`DecisionTreeRegressor`.

```{admonition} Question
Is the decision tree model better in terms of $R^2$ score than the linear
regression?

- a) Yes
- b) No

_Select a single answer_
```

+++

Instead of using the default parameter for decision tree regressor, we will
optimize the depth of the tree. Using a grid-search with a 10-fold cross-validation, answer to the questions below. Vary the `max_depth` from
1 level up to 15 levels.

```{admonition} Question
What the optimal tree depth for the current problem?

- a) The optimal depth is ranging from 3 to 5
- b) The optimal depth is ranging from 5 to 8
- c) The optimal depth is ranging from 8 to 11
- d) The optimal depth is ranging from 11 to 15

_Select a single answer_
```

+++

```{admonition} Question
A tree with an optimal depth is performing:

- a) better than a linear model
- b) equally to a linear model
- c) worse than a linear model

_Select a single answer_
```

+++

Instead of using only the numerical value mentioned above, use the entire
dataset available in `data`. Create a preprocessor by dealing separately with
the numerical and categorical columns. For the sake of simplicity, we will
define the categorical columns as the columns with an `object` data type.

**Do not optimize the `max_depth` parameter for this exercise.**

```{admonition} Question
Are the performance in terms of $R^2$ better by incorporating the categorical
features in comparison with the previous tree with the optimal depth?

- a) No the statistical performance are the same: ~0.7
- b) The statistical performance is slightly better: ~0.72
- c) The statistical performance is better: ~0.74

_Select a single answer_
```
