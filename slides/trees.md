class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Decisions trees

This lesson covers decision tree and random forest. These are robust models for both rergession and classification.

<img src="../scikit-learn-logo.svg">

???

Decision tree, random forest


---
# Outline

* What is a decision tree?
* For classification & rergession
* Decision tree and random forest
* How to avoid overfitting?


---
# Example
.shift-left[<img src="../figures/tree_example.svg" width="80%">]

???
A decision tree is a set of rules, combined in a hierarchical manner.

In this example, if a new point have to be classified :
- we will first check the age feature, if it is lower than 28.5, we shall classified it as "low income".
- Otherwise, depending of the hours per week feature, we will classified it as low or high income.

---
# Classification
.pull-left.shift-left[<img src="../figures/tree2D_1split.svg" width="100%">]

.pull-right[<img src="../figures/tree_blue_orange1.svg" width="100%">]

???
Each split shall maximize the information gain. This will be define precisely in the corresponding notebook.

---
# Classification
.pull-left.shift-left[<img src="../figures/tree2D_2split.svg" width="100%">]

.pull-right[<img src="../figures/tree_blue_orange2.svg" width="100%">]
 
???
We can expand a leaf to refine the decision

---
# Classification
.pull-left.shift-left[<img src="../figures/tree2D_3split.svg" width="100%">]
 
.pull-right[<img src="../figures/tree_blue_orange3.svg" width="100%">]
 
???
In this example, after two split, we obtain pure leaf. 
i.e. in each leaf, there is only one class. 

---
# Regression
<img src="../figures/tree_regression1.svg" width="100%">
???
Decision tree can also fit regression problem. 

---
# Regression
<img src="../figures/tree_regression2.svg" width="100%">
???

It will arrange the split w.r.t. the value of *x*.

---
# Regression
<img src="../figures/tree_regression3.svg" width="100%">
???

But it can also overfit.
Controling the depth allow to control the overfit.


---
# Boosting
.pull-left[<img src="../figures/boosting0.svg" width="100%">]

???
Here we have a classification task. 

---
# Boosting
.pull-left[<img src="../figures/boosting1.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees1.svg" width="100%">]
???
A first tree start to separate circle from square
---
# Boosting
.pull-left[<img src="../figures/boosting2.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees2.svg" width="100%">]

???
The second tree refine the first model
---
# Boosting
.pull-left[<img src="../figures/boosting3.svg" width="100%">]
.pull-right[<img src="../figures/boosting_trees3.svg" width="100%">]
???
A third tree keep refining the model.

---
# Bagging
.pull-left[<img src="../figures/bagging0.svg" width="100%">]
.pull-right[<img src="../figures/bagging.svg" width="120%">]
???

---
# Bagging
.pull-left[<img src="../figures/bagging0.svg" width="100%">]
.pull-right[<img src="../figures/bagging_line.svg" width="120%">]

.pull-right[<img src="../figures/bagging_trees.svg" width="120%">]
???

---
# Bagging
.pull-left[<img src="../figures/bagging0_cross.svg" width="100%">]
.pull-right[<img src="../figures/bagging_cross.svg" width="120%">]

.pull-right[<img src="../figures/bagging_trees_predict.svg" width="120%">]
 
.pull-right[<img src="../figures/bagging_vote.svg" width="120%">]
???



---
# Take away

* `max_depth` parameter can prevent overfiting
* `Random Forest` use bagging over decision tree
