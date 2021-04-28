class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Decision Trees

For regression and classification.

<img src="../figures/scikit-learn-logo.svg">

???

Decision tree are built as a set of rules for both
classification and regression problems.

These are the building blocks for more elaborate models such
as _random forest_ and _gradient boosting trees_, as we will see.

---

# Outline

- What is a decision tree?
- For classification & regression
- Impact of the tree depth on overfitting

---

class: center, middle

# What is a decision tree?

How does it build classification and regression models?

???

First, let us develop a bit the intuitions on what is a decision tree,
and how it can form the basis of classification and regression models.

---

# A example of a decision tree

.shift-up-more.shift-left[<img src="../figures/tree_example.svg" width="100%">]

???
A decision tree is a set of rules, combined in a hierarchical manner.

In this example, if a new point has to be classified:

- we will first check the age feature, if it is lower than 28.5, we shall classify it as "low income".
- Otherwise, depending on the hours per week feature, we will classify it as low or high income.

---

# Classification with a decision tree

.pull-left.shift-left[<img src="../figures/tree2D_1split.svg" width="100%">]

.pull-right[<img src="../figures/tree_blue_orange1.svg" width="100%">]

???
Each split shall maximize the "information gain".
That will be defined precisely in the following notebook.

---

# Classification with a decision tree

.pull-left.shift-left[<img src="../figures/tree2D_2split.svg" width="100%">]

.pull-right[<img src="../figures/tree_blue_orange2.svg" width="100%">]

???

We can incrementally expand any leaf to refine the decision function.
At each step, the leaf focuses on a smaller subregion of the space.

---

# Classification with a decision tree

.pull-left.shift-left[<img src="../figures/tree2D_3split.svg" width="100%">]

.pull-right[<img src="../figures/tree_blue_orange3.svg" width="100%">]

???
In this example, after two splits, we obtain pure leaves.
i.e. in each leaf, there is only one class.
The max depth here is equal to 2. We do not need to go deeper.

---

# Regression with a decision tree

<img src="../figures/tree_regression1.svg" width="100%">

???
Decision trees can also fit regression problems.

---

# Regression with a decision tree

<img src="../figures/tree_regression2.svg" width="100%">

???
It will arrange the split w.r.t. the value of _x_.
Without making any assumption on any distribution.
It is a non-parametric model.

---

# Regression with a decision tree

<img src="../figures/tree_regression3.svg" width="100%">
???

However, it can also overfit.
Controlling the depth here allows to control the overfitting.

---
class: split-3columns
# Tree: underfit / overfit tradeoff

.column[
<img src="../figures/dt_underfit.svg" width="100%">
]

--

.column[
<img src="../figures/dt_fit.svg" width="100%">
]

--

.column[
<img src="../figures/dt_overfit.svg" width="100%">
]

--

.column[
.center[Underfitting]

.center.small[`max_depth` too small]
]

.column[
.center[Best trade-off]
]

.column[
.center[Overfitting]

.center.small[`max_depth` too large]
]

---

# Take home messages

- Sequence of simple decision rules:

  one feature and one threshold at a time

--

- No scaling required for numerical features

--

- `max_depth` controls the trade-off between underfitting and overfitting

--

- Mostly useful as a building block for ensemble models
  - Random Forests
  - Gradient Boosting Decision Trees
