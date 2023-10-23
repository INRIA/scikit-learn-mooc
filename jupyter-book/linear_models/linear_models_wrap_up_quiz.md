# 🏁 Wrap-up quiz 4

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

Start by fitting a ridge regressor (`sklearn.linear_model.Ridge`) fixing the
penalty `alpha` to 0 to not regularize the model.
Use a 10-fold cross-validation and pass the argument `return_estimator=True` in
`sklearn.model_selection.cross_validate` to access all fitted estimators fitted
on each fold. As discussed in the previous notebooks, use an instance of
`sklearn.preprocessing.StandardScaler` to scale the data before passing it to
the regressor.

```{admonition} Question
How large is the largest absolute value of the weight (coefficient)
in this trained model?

- a) Lower than 1.0 (1e0)
- b) Between 1.0 (1e0) and 100,000.0 (1e5)
- c) Larger than 100,000.0 (1e5)

_Select a single answer_

Hint: Note that the estimator fitted in each fold of the cross-validation
procedure is a pipeline object. To access the coefficients of the
`Ridge` model at the last position in a pipeline object, you can
use the expression `pipeline[-1].coef_` for each pipeline object
fitted in the cross-validation procedure. The `-1` notation is a
negative index meaning "last position".
```

+++

Repeat the same experiment by fitting a ridge regressor
(`sklearn.linear_model.Ridge`) with the default parameter (i.e. `alpha=1.0`).

```{admonition} Question
How large is the largest absolute value of the weight (coefficient)
in this trained model?

- a) Lower than 1.0
- b) Between 1.0 and 100,000.0
- c) Larger than 100,000.0

_Select a single answer_
```

+++

```{admonition} Question
What are the two most important features used by the ridge regressor? You can
make a box-plot of the coefficients across all folds to get a good insight.

- a) `"MiscVal"` and `"BsmtFinSF1"`
- b) `"GarageCars"` and `"GrLivArea"`
- c) `"TotalBsmtSF"` and `"GarageCars"`

_Select a single answer_
```

+++

Remove the feature `"GarageArea"` from the dataset and repeat the previous
experiment.

```{admonition} Question
What is the impact on the weights of removing `"GarageArea"` from the dataset?

- a) None
- b) Completely changes the order of the most important features
- c) Decreases the standard deviation (across CV folds) of the `"GarageCars"` coefficient

_Select all answers that apply_
```

+++

```{admonition} Question
What is the main reason for observing the previous impact on the most
important weight(s)?

- a) Both garage features are correlated and are carrying similar information
- b) Removing the "GarageArea" feature reduces the noise in the dataset
- c) Just some random effects

_Select a single answer_
```

+++

Now, we will search for the regularization strength that maximizes the
generalization performance of our predictive model. Fit a
[`sklearn.linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
instead of a `Ridge` regressor on the numerical data without the `"GarageArea"`
column. Pass `alphas=np.logspace(-3, 3, num=101)` to explore the effect of
changing the regularization strength.

```{admonition} Question
What is the effect of tuning `alpha` on the variability of the weights of the
feature `"GarageCars"`? Remember that the variability can be assessed by
computing the standard deviation.

- a) The variability does not change after tuning `alpha`
- b) The variability decreased after tuning alpha
- c) The variability increased after tuning alpha

_Select a single answer_
```

+++

Check the parameter `alpha_` (the regularization strength) for the different
ridge regressors obtained on each fold.

```{admonition} Question
In which range does `alpha_` fall into for most folds?

- a) between 0.1 and 1
- b) between 1 and 10
- c) between 10 and 100
- d) between 100 and 1000

_Select a single answer_
```

+++

So far we only used the list of `numerical_features` to build the predictive
model. Now create a preprocessor to deal separately with the numerical and
categorical columns:

- categorical features can be selected if they have an `object` data type;
- use an `OneHotEncoder` to encode the categorical features;
- numerical features should correspond to the `numerical_features` as defined
  above. This is a subset of the features that are not an `object` data type;
- use an `StandardScaler` to scale the numerical features.

The last step of the pipeline should be a `RidgeCV` with the same set of `alphas`
to evaluate as previously.

```{admonition} Question
By comparing the cross-validation test scores fold-to-fold for the model with
`numerical_features` only and the model with both `numerical_features` and
`categorical_features`, count the number of times the simple model has a better
test score than the model with all features. Select the range which this number
belongs to:

- a) [0, 3]: the simple model is consistently worse than the model with all features
- b) [4, 6]: both models are almost equivalent
- c) [7, 10]: the simple model is consistently better than the model with all features

_Select a single answer_
```

+++

In this Module we saw that non-linear feature engineering may yield a more
predictive pipeline, as long as we take care of adjusting the regularization to
avoid overfitting.

Try this approach by building a new pipeline similar to the previous one but
replacing the `StandardScaler` by a `SplineTransformer` (with default
hyperparameter values) to better model the non-linear influence of the
numerical features.

Furthermore, let the new pipeline model feature interactions by adding a new
`Nystroem` step between the preprocessor and the `RidgeCV` estimator. Set
`kernel="poly"`, `degree=2` and `n_components=300` for this new feature
engineering step.

```{admonition} Question
By comparing the cross-validation test scores fold-to-fold for the model with
both `numerical_features` and `categorical_features`, and the model that
performs non-linear feature engineering; count the number of times the
non-linear pipeline has a better test score than the model with simpler
preprocessing. Select the range which this number belongs to:

- a) [0, 3]: the new non-linear pipeline is consistently worse than the previous pipeline
- b) [4, 6]: both models are almost equivalent
- c) [7, 10]: the new non-linear pipeline is consistently better than the previous pipeline

_Select a single answer_
```
