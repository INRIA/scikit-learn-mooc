class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Linear Model

This lesson covers the linear models. These are basic models, easy to understand and fast to train

<img src="../scikit-learn-logo.svg">

???

Linear models are easy to understand and fast to train
They give us a baseline

plan : 
Logistic Regression, linear regression, classification vs regression, multi-class, linear separability. Pros and cons L1 and L2 penalty for linear models Learning curves and validation curves (video: how to read curves)

---
# Outline

* Linear regression
 - plot 2D
 - loss (with plot)
 - 3-dim and n-dim
* Logistic regression
 - plot 2D
* multi class
* linear separability
* L1 and L2 penalty

---
# Linear regression

Here we have to predict the value of the target **y** given some observation (explanatory variable) **X**. For illustration purpose, we will consider that the observation is only one dimensional.

.shift-left.pull-right[<img src="../figures/linear_data.svg" width="110%">]

???
Here the target value is expected to be a linear combination of the features
 
---
# Linear regression

Once fitted :y = a*x + b

.shift-left.pull-left[<img src="../figures/linear_fit.svg" width="100%">]

???
Here the target value is expected to be a linear combination of the features
 
---
# End

???

.shift-left.pull-left[<img src="../figures/linear_ols.svg" width="110%">]

.pull-right[<img src="../figures/linear_splines.svg" width="110%">]

.shift-left.pull-left[<img src="../figures/linear_ols_test.svg" width="110%">]
.pull-right[<img src="../figures/linear_splines_test.svg" width="110%">]
.centered.reversed[**On new data**]

.shift-left.pull-left[<img src="../figures/ols_simple_test.svg" width="110%">]
.pull-right[<img src="../figures/splines_cubic_test.svg" width="110%">]
.centered[A harder example]
