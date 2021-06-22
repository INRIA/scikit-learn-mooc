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

Start by fitting a linear regression (`sklearn.linear_model.LinearRegression`).
Use a 10-fold cross-validation and pass the argument `return_estimator=True` in
`sklearn.model_selection.cross_validate` to access all fitted estimators fitted
on each fold. As we saw in the previous notebooks, you will have to use a
`sklearn.preprocessing.StandardScaler` to scale the data before passing it to
the regressor. Also, some missing data are present in the different columns.
You can use a `sklearn.impute.SimpleImputer` with the default parameters to
impute missing data. Thus, you can create a model that will **pipeline the
scaler, followed by the imputer, followed by the linear regression**.

```{admonition} Question
What is the order of magnitude of the extremum weight values over all the features:

- a) 1e4
- b) 1e6
- c) 1e18

_Select a single answer_
```

+++

Repeat the same experiment by fitting a ridge regressor
(`sklearn.linear_model.Ridge`) with the default parameter.

```{admonition} Question
What magnitude of the extremum weight values for all features?

- a) 1e4
- b) 1e6
- c) 1e18

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
- b) Change completely the order of the feature importance
- c) The variability of the most important feature reduced

_Select a single answer_
```

+++

```{admonition} Question
What is the reason for observing the previous impact on the most important
weight?

- a) Both features are correlated and are carrying similar information
- b) Removing a feature reduce the noise in the dataset
- c) Just some random effects

_Select a single answer_
```

+++

Now, we will search for the regularization strength that will maximize the
statistical performance of our predictive model. Fit a
[`sklearn.linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
instead of a `Ridge` regressor pass `alphas=np.logspace(-1, 3, num=30)` to
explore the effect of changing the regularization strength.

```{admonition} Question
Are there major differences regarding the most important weights?

- a) Yes, the weights order is completely different
- b) No, the weights order is very similar

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

Now, we will tackle a classification problem instead of a regression problem.
Load the Adult Census dataset with the following snippet of code and we will
work only with **numerical features**.

```py
adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
data = adult_census.select_dtypes(["integer", "floating"])
data = data.drop(columns=["education-num"])
```

```{admonition} Question
How many numerical features are present in the dataset contained in the
variable `data`?

- a) 3
- b) 4
- c) 5

_Select a single answer_
```

+++

```{admonition} Question
Are there any missing values in the dataset contained in the variable `data`?

- a) Yes
- b) No

_Select a single answer_

Hint: you can use `df.info()` to get information regarding each column.
```

+++

Fit a `sklearn.linear_model.LogisticRegression` classifier using a 10-fold
cross-validation to assess the performance. Since we are dealing with a linear
model, do not forget to scale the data with a `StandardScaler` before training
the model.

```{admonition} Question
On average, how much better/worse/similar is the logistic regression to a dummy
classifier that would predict the most frequent class? We will consider an
increase or decrease of the accuracy score.

- a) Worse than a dummy classifier with a decrease of 0.04
- b) Similar to a dummy classifier
- c) Better than a dummy classifier with an increase 0.04

_Select a single answer_
```

+++

```{admonition} Question
What is the most important feature seen by the logistic regression?

- a) `"age"`
- b) `"capital-gain"`
- c) `"capital-loss"`
- d) `"hours-per-week"`

_Select a single answer_
```

+++

Now, we will work with **both numerical and categorical features**. You can
load Adult Census with the following snippet:

```py
adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
data = adult_census.drop(columns=["class", "education-num"])
```

```{admonition} Question
Are there missing values in this dataset?

- a) Yes
- b) No

_Select a single answer_

Hint: you can use `df.info()` to get information regarding each column.
```

+++

Create a predictive model where the categorical data should be one-hot encoded,
the numerical data should be scaled, and the predictor used should be a
logistic regression classifier.

```{admonition} Question
On average, what is the increase in terms of accuracy by using the categorical
features?

- a) It gives similar results
- b) It increases the statistical performance by 0.025
- c) it increases the statistical performance by 0.05
- d) it increases the statistical performance by 0.075
- e) it increases the statistical performance by 0.1

_Select a single answer_
```

+++

For the following questions, you can use the following snippet to get the
feature names after the preprocessing performed.

```py
preprocessor.fit(data)
feature_names = (preprocessor.named_transformers_["onehotencoder"]
                             .get_feature_names(categorical_columns)).tolist()
feature_names += numerical_columns
```

There is as many feature names as coefficients in the last step of your
predictive pipeline.

```{admonition} Question
What are the two most important features used by the logistic regressor?

- a) `"hours-per-week"` and `"native-country_Columbia"`
- b) `"workclass_?"` and `"naitive_country_?"`
- c) `"capital-gain"` and `"education_Doctorate"`

_Select a single answer_
```

+++

```{admonition} Question
What is the effect of decreasing the `C` parameter on the coefficients?

- a) shrinking the magnitude of the weights towards zeros
- b) increasing the magnitude of the weights
- c) reducing the weights' variance
- d) increasing the weights' variance
- e) it has no influence on the weights' variance

_Select several answers_
```
