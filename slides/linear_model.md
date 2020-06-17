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

We look for the best fit. 
Learning w_0 and w_1 such that 
\sum_i | y_i - w_o + w_1 * x_i|^2 is minimal


.shift-left.pull-left[<img src="../figures/linear_fit.svg" width="100%">]

???

---
# Error in linear regression

For each sample x_i, we have an error which correspond to |y_i - ŷ_i|^2
That correspond to the red bar in the figure below
.shift-left.pull-left[<img src="../figures/linear_fit.svg" width="100%">]

???

---
# Linear regerssion in higher dimension

If **X** has two dimensions, we obtain a plot like that:

---
# Logistic regression

With Logistic regression, we learn a linear model for classification (and not regresion).
So **y** is either +1 or -1

.shift-left.pull-left[<img src="../figures/categorical.svg" width="110%">]


---
# Logistic regression

sigmoïd: sigma(x) = 1 / (1 + exp(-x))

.shift-left.pull-left[<img src="../figures/logistic_color.svg" width="110%">]


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
