# ✅ Quiz M7.05

```{admonition} Question
What is the default score in scikit-learn when using a regressor?

- a) $R^2$
- b) mean absolute error
- c) median absolute error

_Select a single answer_
```

+++

```{admonition} Question
If we observe that the values returned by
`cross_val_scores(model, X, y, scoring="r2")` increase after changing the model
parameters, it means that the latest model:

- a) generalizes better
- b) generalizes worse

_Select a single answer_
```

+++

```{admonition} Question
If all the values returned by
`cross_val_score(model_A, X, y, scoring="neg_mean_squared_error")`
are strictly lower than those returned by
`cross_val_score(model_B, X, y, scoring="neg_mean_squared_error")`
it means that `model_B` generalizes:

- a) better than `model_A`
- b) worse than `model_A`

Hint: Remember that `"neg_mean_squared_error"` is an alias for the negative of
the Mean Squared Error.

_Select a single answer_
```

+++

```{admonition} Question
Values returned by `cross_val_scores(model, X, y, scoring="neg_mean_squared_error")`
are:

- a) guaranteed to be positive or zero
- b) guaranteed to be negative or zero
- c) can be either positive or negative depending on the data

_Select a single answer_
```
