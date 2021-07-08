# ✅ Quiz M3.01

```{admonition} Question
Which parameters below are hyperparameters of `HistGradientBosstingClassifier`?

- a) `C`
- b) `max_leaf_nodes`
- c) `verbose`
- d) `classes_`
- e) `learning_rate`
```

+++

````{admonition} Question
Given `model` defined by:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

how do you get the value of the `C` parameter?
- a) `model.get_parameters()['C']`
- b) `model.get_params()['C']`
- c) `model.get_params('C')`
- d) `model.get_params['C']`
````

+++

````{admonition} Question
Given `model` defined by:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

how do you set the value of the `C` parameter to `5`?
- a) `model.set_params('C', 5)`
- b) `model.set_params({'C': 5})`
- c) `model.set_params()['C'] = 5`
- d) `model.set_params(C=5)`
````

+++

````{admonition} Question
Given `model` defined by:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

how do you set the value of the `C` parameter of the `LogisticRegression` component to 5:
- a) `model.set_params(C=5) `
- b) `model.set_params(logisticregression__C=5)`
- c) `model.set_params(classifier__C=5) `
- d) `model.set_params(classifier--C=5)`
````
