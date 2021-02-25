# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `blood_transfusion.csv`.

```py
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]
```

In this dataset, the column `"Class"` is the target vector containing the
labels that our model should predict.

For all the questions below, make a cross-validation evaluation using a
10-fold cross-validation strategy.

Evaluate the performance of a `DummyClassifier` that always predict the most
frequent class seen during the training. Be aware that you can pass a list
of score to compute in `cross_validate` by setting the parameter `scoring`.

```{admonition} Question
What the accuracy of this dummy classifier?

- a) ~0.5
- b) ~0.62
- c) ~0.75

_Select a single answer_
```

+++

```{admonition} Question
What the balanced accuracy of this dummy classifier?

- a) ~0.5
- b) ~0.62
- c) ~0.75

_Select a single answer_
```

+++

Replace the `DummyClassifier` by a `DecisionTreeClassifier` and check the
statistical performance to answer the question below.

```{admonition} Question
Is a single decision classifier better than a dummy classifier (at least an
increase of 4%) in terms of balanced accuracy?

- a) Yes
- b) No

_Select a single answer_
```

+++

Evaluate the performance of a `RandomForestClassifier` using 300 trees.

```{admonition} Question
Is random forest better than a dummy classifier (at least an increase of 4%)
in terms of balanced accuracy?

- a) Yes
- b) No

_Select a single answer_
```

+++

Compare a `GradientBoostingClassifier` and a `RandomForestClassifier` with both
300 trees. Evaluate both models with a 10-fold cross-validation and repeat the
experiment 10 times.

```{admonition} Question
In average, is the gradient boosting better than the random forest?

- a) Yes
- b) No
- c) Equivalent

_Select a single answer_
```

+++

Evaluate the performance of a `HistGradientBoostingClassifier`. Enable
early-stopping and add as many trees as needed.

```{admonition} Question
Is histogram gradient boosting a better classifier?

- a) Histogram gradient boosting is the best estimator
- b) Histogram gradient boosting is better than random forest by worse than
  the exact gradient boosting
- c) Histogram gradient boosting is better than the exact gradient boosting but
  worse than the random forest
- d) Histogram gradient boosting is the worst estimator

_Select a single answer_
```

+++

```{admonition} Question
With the early stopping activated, how many trees in average the
`HistGradientBoostingClassifier` needed to converge?

- a) ~30
- b) ~100
- c) ~150
- d) ~200
- e) ~300

_Select a single answer_
```

+++

Use imbalanced-learn and the class `BalancedBaggingClassifier`. Provide the
previous histogram gradient boosting as a base estimator and train this bagging
classifier with 50 estimators.

```{admonition} Question
What is a [`BalancedBaggingClassifier`](https://imbalanced-learn.org/stable/ensemble.html#bagging)?

- a) Is a classifier that make sure that each tree leaves belong to the same
  depth level
- b) Is a classifier that explicitly maximizes the balanced accuracy score
- c) Equivalent to a `BaggingClassifier` with a resampling of each bootstrap
     sample to contain a many samples from each class.

_Select a single answer_
```

+++

```{admonition} Question
Is the balanced accuracy of the `BalancedBaggingClassifier` is
_choose an answer_ than an `HistGradientBoostingClassifier` alone?

- a) Worse
- b) Better
- c) Equivalent

_Select a single answer_
```
