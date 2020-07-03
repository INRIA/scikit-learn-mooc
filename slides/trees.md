class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Decisions trees

This lesson covers decisions trees and random forest. These are robust models for both rergession and classification.

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

???
Each split maximize the information

---
# Classification
.pull-left.shift-left[<img src="../figures/tree2D_2split.svg" width="100%">]
???
In this example, after two split, we obtain pur leaf. 
In each leaf, there is only one class. 

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
