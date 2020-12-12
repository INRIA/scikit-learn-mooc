# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `blood_transfusion.csv`. The column `"Class"` is the target
vector containing the labels that our model to predict.

In all the experiments, use a 10-Fold cross-validation.

Evaluate the performance of a `DummyClassifier` that always predict the most
frequent class seen during the training.

```{admonition} Question
What the accuracy of this dummy classifier?
_Select a single answer_

- a) ~0.5
- b) ~0.62
- c) ~0.75
```

```{admonition} Question
What the balanced accuracy of this dummy classifier?
_Select a single answer_

- a) ~0.5
- b) ~0.62
- c) ~0.75
```

Evaluate the performance of a single `DecisionTreeClassifier`.

```{admonition} Question
Is a single decision classifier better than a dummy classifier (at least an
increase of 2%)?
_Select a single answer_

- a) Yes
- b) No
```

Evaluate the performance of a `RandomForestClassifier` using 300 trees.

```{admonition} Question
Is random forest better than a dummy classifier (at least an increase of 2%)?
_Select a single answer_

- a) Yes
- b) No
```

Evaluate the performance of a `GradientBoostingClassifier` using 300 trees.
Repeat the evaluation 10 times.

```{admonition} Question
In average, is the gradient boosting better than the random forest?
_Select a single answer_

- a) Yes
- b) No
- c) Equivalent
```

Evaluate the performance of a `HistGradientBoostingClassifier`. Enable the
early-stopping and add as much trees as needed.

```{admonition} Question
Is histogram gradient boosting a better classifier?
_Select a single answer_

- a) Histogram gradient boosting is the best estimator
- b) Histogram gradient boosting is better than random forest by worse than
  the exact gradient boosting
- c) Histogram gradient boosting is better than the exact gradient boosting but
  worse than the random forest
- d) Histogram gradient boosting is the worst estimator
```

Use imbalanced-learn and the class `BalancedBaggingClassifier`. Provide the
previous histogram gradient boosting.

```{admonition} Question
What is a `BalancedBaggingClassifier`?
_Select a single answer_

- a) Is a classifier that make sure that each tree leaves belong to the same
  depth level
- b) Is a classifier that maximizes the balanced accuracy score
- c) Equivalent to a `BaggingClassifier` where each bootstrap will
  contain as much sample from each class
```

```{admonition} Question
Is this balanced bagging classifier improving the balanced accuracy compared
to the best classifier observed in the previous experiment?
_Select a single answer_

- a) Yes
- b) No
- c) Equivalent
```
