# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `ames_housing_no_missing.csv` with the following command:

```python
import pandas as pd

ames_housing = pd.read_csv("../datasets/ames_housing_no_missing.csv")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]
```

`ames_housing` is a pandas dataframe. The column "SalePrice" contains the
target variable.

To simplify this exercise, we will only used the numerical features defined
below:

```python
numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]
```

We will compare the generalization performance of a decision tree and a linear
regression. For this purpose, we will create two separate predictive models
and evaluate them by 10-fold cross-validation.

Thus, use `sklearn.linear_model.LinearRegression` and
`sklearn.tree.DecisionTreeRegressor` to create the model. Use the default
parameters for both models.

Be aware that a linear model requires to scale the data. You can use a
`sklearn.preprocessing.StandardScaler`.

```{admonition} Question
Is the decision tree model better in terms of $R^2$ score than the linear
regression?

- a) Yes
- b) No

_Select a single answer_
```

+++

Instead of using the default parameter for decision tree regressor, we will
optimize the depth of the tree. Using a grid-search
(`sklearn.model_selection.GridSearchCV`) with a 10-fold cross-validation,
answer to the questions below. Vary the `max_depth` from 1
level up to 15 levels.

```{admonition} Question
What is the optimal tree depth for the current problem?

- a) The optimal depth is ranging from 3 to 5
- b) The optimal depth is ranging from 5 to 8
- c) The optimal depth is ranging from 8 to 11
- d) The optimal depth is ranging from 11 to 15

_Select a single answer_
```

+++

```{admonition} Question
A tree with an optimal depth has a score of:

a) ~0.74 and is better than the linear model
b) ~0.72 and is equal to the linear model
c) ~0.7 and is worse than the linear model

_Select a single answer_
```

+++

Instead of using only the numerical dataset you will now use the entire
dataset available in the variable `data`.

Create a preprocessor by dealing separately with the numerical and categorical
columns. For the sake of simplicity, we will assume the following:

- categorical columns can be selected if they have an `object` data type;
- numerical columns can be selected if they do not have an `object` data type.
  It will be the complement of the numerical columns.

**Do not optimize the `max_depth` parameter for this exercise.** Keep the
default value (`None`) for this parameter.

**Fix the random state of the tree by passing the parameter `random_state=0`**

```{admonition} Question
Are the performance in terms of $R^2$ better by incorporating the categorical
features in comparison with the previous tree with the optimal depth?

- a) No, the generalization performance is the same: ~0.7
- b) The generalization performance is slightly better: ~0.72
- c) The generalization performance is better: ~0.74

_Select a single answer_
```
