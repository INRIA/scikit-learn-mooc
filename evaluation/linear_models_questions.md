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

Start by fitting a linear regression. Use a 10-Fold cross-validation and pass
the argument `return_estimator=True` to access to all estimators fitted for
each fold.

```{admonition} Question
What magnitude of the extremum weight values for all features:
_Select a single answer_

- a) 1e4
- b) 1e6
- c) 1e18
```

Now, use a ridge regressor instead of a linear regression.

```{admonition} Question
What magnitude of the extremum weight values for all features?
_Select a single answer_

- a) 1e4
- b) 1e6
- c) 1e18
```

```{admonition} Question
What are the two most important features used by the ridge regressor?
_Select a single answer_

- a) `"MiscVal"` and `"BsmtFinSF1"`
- b) `"GarageCars"` and `"GrLivArea"`
- c) `"TotalBsmtSF"` and `"GarageCars"`
```

Remove the feature `"GarageArea"` from the dataset and repeat the previous
experiment.

```{admonition} Question
What is the impact on the weights of removing `"GarageArea"` from the dataset?
_Select a single answer_

- a) None
- b) Change completely the order of the feature importance
- c) The variability of the most important feature reduced
```

```{admonition} Question
What is the reason for observing the previous impact on the most important
weight?
_Select a single answer_

- a) Both features were correlated and were carrying similar information
- b) ...
- c) ...
```

Instead of using a ridge regressor, use a `RidgeCV` where the goal will be to
find the optimum l2 penalty with an internal cross-validation. You can provide
a range of `alphas` as `np.logspace(-1, 3, num=30)`

```{admonition} Question
Is there major differences regarding the most important weights?
_Select a single answer_

- a) Yes, the weights order is completely different
- b) No, the weights order is very similar
```

Check the parameter `alpha_` of the different ridge regressor

```{admonition} Question
In general what is the optimal l2 penalty strength?
_Select a single answer_

- a) between 0.1 and 1
- b) between 1 and 10
- c) between 10 and 100
- d) between 100 and 1000
```
