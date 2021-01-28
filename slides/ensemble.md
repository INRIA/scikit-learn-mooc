class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Ensemble of tree-based models

This lesson covers models based on the aggregation of decision trees.
They are known as gradient-boosting and random forest

These are robust models for both regression and classification.

<img src="../figures/scikit-learn-logo.svg">

???
Decision trees are built as a set of rules for both 
classification and regression problems.

These are the building blocks for more elaborate model such 
as *random forest* and *gradient boosting trees*, as we will see.

---

# Outline

- Bagging
- Boosting

---

# Bagging

<img src="../figures/bagging_reg_data.svg" width="50%">

---

# Bagging

.shift-up-less[
<img src="../figures/bagging_reg_grey.svg" width="120%">
]
.pull-left[

- Select multiple random subsets of the data
  ]

---

# Bagging

.shift-up-less[
<img src="../figures/bagging_reg_grey_fitted.svg" width="120%">
]

.pull-left[

- Select multiple random subsets of the data

- Fit one model on each
  ]

---

# Bagging

.shift-up-less[
<img src="../figures/bagging_reg_grey_fitted.svg" width="120%">
]

.pull-left[

- Select multiple random subsets of the data

- Fit one model on each

- Average predictions
  ]

.pull-right[
<img src="../figures/bagging_reg_blue.svg" width="80%">
]

???

In bagging, we will construct deep trees in parallel.

Each tree will be fitted on a sub-sampling from the initial data.
i.e. we will only consider a random part of the data to build each model.

When we have to classify a new point, we will aggregate the prediction of every model by a voting scheme.

---

# Bagging trees: random forests

.pull-left[<img src="../figures/bagging0.svg" width="100%">]
.pull-right[<img src="../figures/bagging.svg" width="120%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import RandomForestClassifier
```

]

???
Here we have a classification task: separating circles from squares.

---

# Bagging trees: random forests

.pull-left[<img src="../figures/bagging0.svg" width="100%">]
.pull-right[<img src="../figures/bagging_line.svg" width="120%">]

.pull-right[<img src="../figures/bagging_trees.svg" width="120%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import RandomForestClassifier
```

]

???

---

# Bagging trees: random forests

.pull-left[<img src="../figures/bagging0_cross.svg" width="100%">]
.pull-right[<img src="../figures/bagging_cross.svg" width="120%">]

.pull-right[<img src="../figures/bagging_trees_predict.svg" width="120%">]

.pull-right[<img src="../figures/bagging_vote.svg" width="120%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import RandomForestClassifier
```

]

???

---

# Boosting

## <img src="../figures/boosting/boosting_iter1.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter_sized1.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter_orange1.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter2.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter_sized2.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter_orange2.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter3.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter_sized3.svg" width="100%">

# Boosting

## <img src="../figures/boosting/boosting_iter_orange3.svg" width="100%">

# Boosting

<img src="../figures/boosting/boosting_iter4.svg" width="100%">

???

---

# Boosting

.pull-left[<img src="../figures/boosting0.svg" width="100%">]

???

---

# Boosting

.pull-left[<img src="../figures/boosting1.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees1.svg" width="100%">]

???
A first shallow tree starts to separate circles from squares. 
Mistakes done by this first tree model shall be corrected 
by a second tree model.

---

# Boosting

.pull-left[<img src="../figures/boosting2.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees2.svg" width="100%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate = .1)
```

]

???
So now, the second tree refines the first tree. 
The final model is a weighted sum of these two trees.

---

# Boosting

.pull-left[<img src="../figures/boosting3.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees3.svg" width="100%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate = .1)
```

]

???
We could continue to refining our ensemble model. 
At each step we focus on mistakes of the previous model.

---

# Take away

- `boosting` fits sequentially shallow trees
- `bagging` fits simultaneously deep trees
