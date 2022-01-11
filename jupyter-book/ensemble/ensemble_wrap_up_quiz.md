# 🏁 Wrap-up quiz

**This quiz requires some programming to be answered.**

Open the dataset `blood_transfusion.csv`.

```python
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]
```

In this dataset, the column `"Class"` is the target vector containing the
labels that our model should predict.

For all the questions below, make a cross-validation evaluation using a
10-fold cross-validation strategy.

Evaluate the performance of a `sklearn.dummy.DummyClassifier` that always
predict the most frequent class seen during the training. Be aware that you can
pass a list of score to compute in `sklearn.model_selection.cross_validate` by
setting the parameter `scoring`.

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

Replace the `DummyClassifier` by a `sklearn.tree.DecisionTreeClassifier` and
check the generalization performance to answer the question below.

```{admonition} Question
Is a single decision classifier better than a dummy classifier, by an increase
of at least 0.04 of the balanced accuracy?

- a) Yes
- b) No

_Select a single answer_
```

+++

Evaluate the performance of a `sklearn.ensemble.RandomForestClassifier` using
300 trees.

```{admonition} Question
Is random forest better than a dummy classifier, by an increase of at least
0.04 of the balanced accuracy?

- a) Yes
- b) No

_Select a single answer_
```

+++

Compare a `sklearn.ensemble.GradientBoostingClassifier` and a
`sklearn.ensemble.RandomForestClassifier` with both 300 trees. To do so, repeat
10 times a 10-fold cross-validation by using the balanced accuracy as metric.
For each of the ten try, compute the average of the cross-validation score
for both models. Count how many times a model is better than the other.

```{admonition} Question
On average, is the gradient boosting better than the random forest?

- a) Yes
- b) No
- c) Equivalent

_Select a single answer_
```

+++

Evaluate the performance of a
`sklearn.ensemble.HistGradientBoostingClassifier`. Enable early-stopping and
add as many trees as needed.

```{admonition} Question
Is histogram gradient boosting a better classifier considering the mean of
the cross-validation test score?

- a) Histogram gradient boosting is the best estimator
- b) Histogram gradient boosting is better than random forest but worse than
  the exact gradient boosting
- c) Histogram gradient boosting is better than the exact gradient boosting but
  worse than the random forest
- d) Histogram gradient boosting is the worst estimator

_Select a single answer_
```

+++

```{admonition} Question
With the early stopping activated, how many trees on average the
`HistGradientBoostingClassifier` needed to converge?

- a) ~30
- b) ~100
- c) ~150
- d) ~200
- e) ~300

_Select a single answer_
```
