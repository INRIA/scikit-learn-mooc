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

# Bagging for classification

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

# Bagging for classification

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

# Bagging for classification

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

# Bagging for regression

<img src="../figures/bagging_reg_data.svg" width="50%">

---

# Bagging for regression

.shift-up-less[
<img src="../figures/bagging_reg_grey.svg" width="120%">
]
.pull-left[

- Select multiple random subsets of the data
  ]

---

# Bagging for regression

.shift-up-less[
<img src="../figures/bagging_reg_grey_fitted.svg" width="120%">
]

.pull-left[

- Select multiple random subsets of the data

- Fit one model on each
  ]

---

# Bagging for regression

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

# Boosting for classification

.pull-left[<img src="../figures/boosting1.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees1.svg" width="100%">]

???
A first shallow tree starts to separate circles from squares.
Mistakes done by this first tree model shall be corrected
by a second tree model.

---

# Boosting for classification

.pull-left[<img src="../figures/boosting2.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees2.svg" width="100%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate=.1)
```

]

???
So now, the second tree refines the first tree.
The final model is a weighted sum of these two trees.

---

# Boosting for regression

.pull-left[<img src="../figures/boosting0.svg" width="100%">]

---

# Boosting for classification

.pull-left[<img src="../figures/boosting3.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees3.svg" width="100%">]

.width65.shift-up-less.centered[

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate=.1)
```

]

???
We could continue to refining our ensemble model.
At each step we focus on mistakes of the previous model.

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter1.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter_sized1.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter_orange1.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter2.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter_sized2.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter_orange2.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter3.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter_sized3.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter_orange3.svg" width="95%">

---

# Boosting for regression

<img src="../figures/boosting/boosting_iter4.svg" width="95%">

---

# Take away

- **bagging** and **random forest** independently fits deep trees
  + each individual tree overfits
  + averaging the individual tree predictions fights overfitting
- **boosting** fits sequentially shallow trees
  + each individual tree underfits
  + sequentially adding trees reduces overfitting
- **gradient boosting** tend to perform slightly better than **bagging** and
  **random forest** and furthermore and shallow trees predict faster.
