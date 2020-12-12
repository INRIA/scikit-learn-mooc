# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `house_prices.yml`. The column `SalePrice` contains the
target. Use only the following numerical features when creating a linear
model:

```
numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
```

Start by fitting a linear regression and a decision tree. Use a 10-Fold
cross-validation. Do not tune the hyperparameters.

```{admonition} Question
Is the decision tree model better in terms of $R^2$ score than the linear
regression?
_Select a single answer_

- a) Yes
- b) No
```

Find the optimal depth of the decision tree using a grid-search within the
cross-validation.

```{admonition} Question
What the optimal tree depth for the current problem?
_Select a single answer_

- a) The optimal depth is ranging from 3 to 5
- b) The optimal depth is ranging from 5 to 8
- c) The optimal depth is above 8
```

```{admonition} Question
A tree with an optimal depth is performing:
_Select a single answer_

- a) better than a linear model
- b) equally to a linear model
- c) worse than a linear model
```

Now, instead of only using the numerical data, we will use both numerical and
categorical data. However, only encode the data with `object` dtype, letting
columns with numbers as is. You don't have to optimize the depth of the tree
for this experiment.

```{admonition} Question
Are the performance in terms of $R^2$ better by incorporating the categorical
features in comparison with the previous tree with the optimal depth?
_Select a single answer_

- a) No the performance are the same: ~0.68
- b) The performance is slightly better: ~0.72
- c) The performance is better: ~0.75
```
