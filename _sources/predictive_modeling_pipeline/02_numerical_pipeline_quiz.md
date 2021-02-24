# ✅ Quiz

```{admonition} Question
Why do we need two sets: a train set and a test set?

- a) to train the model faster
- b) to validate the model on unseen data
- c) to improve the accuracy of the model
```

+++

```{admonition} Question
The generalization performance of a scikit-learn model can be evaluated by:

- a) calling `fit` to train the model on the training set, `predict` on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function
- b) calling `fit` to train the model on the training set, `score` to compute the score on the test set
- c) calling `cross_val_score` by passing the model and the data
```

+++

```{admonition} Question
When calling `cross_val_score(estimator, X, y, cv=5)`, the following happens:

- a) X and y are internally split five times with non-overlapping test sets
- b) `estimator.fit` is called 5 times on the full X and y
- c) `estimator.fit` is called 5 times, each time on a different training set
- d) 5 scores, computed on the 5 train sets, are returned
- e) 5 scores, computed on the 5 test sets, are returned
- f) the average of 5 scores is returned
- g) the 5 trained estimators are returned
```

+++

```{admonition} Question
The `StandardScaler` transformer:

- a) transform the features so that they have similar ranges
- b) transform the features to lie in the [0.0, 1.0] range
- c) transform positive-only features into negative or positive values
- d) can help logistic regression converge faster (fewer iterations)
- e) deal with the problem of missing values
```

+++

```{admonition} Question
Cross-validation allows us to:

- a) train the model faster
- b) measure the generalization performance of the model
- c) reach better generalization performance
- d) estimate the variability of the generalization score
```

+++

```{admonition} Question
`make_pipeline` (as well as `Pipeline`) is used to:

- a) run cross-validation
- b) combine one or several transformers and a predictor
- c) try several models at the same time
- d) avoid over-fitting
```
