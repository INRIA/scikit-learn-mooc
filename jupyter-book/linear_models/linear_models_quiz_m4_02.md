# ✅ Quiz M4.02

```{admonition} Question

Let us consider a pipeline that combines a polynomial feature extraction of
degree 2 and a linear regression model. Let us assume that the linear regression
coefficients are all non-zero and that the dataset contains a single feature.
Is the prediction function of this pipeline a straight line?

- a) yes
- b) no

_Select a single answer_
```

+++

```{admonition} Question
Fitting a linear regression where `X` has `n_features` columns and the target
is a single continuous vector, what is the respective type/shape of `coef_`
and `intercept_`?

- a) it is not possible to fit a linear regression in dimension higher than 2
- b) array of shape (`n_features`,) and a float
- c) array of shape (1, `n_features`) and an array of shape (1,)

_Select a single answer_
```

+++

```{admonition} Question
Combining (one or more) feature engineering transformers in a single pipeline:

- a) increases the expressivity of the model
- b) ensures that models extrapolate accurately regardless of the distribution of the data
- c) may require tuning additional hyperparameters
- d) inherently prevents any underfitting

_Select all answers that apply_
```
