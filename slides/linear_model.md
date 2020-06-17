class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Linear Model

This lesson covers the linear models. These are basic models, easy to understand and fast to train

<img src="../scikit-learn-logo.svg">

???

Linear models are easy to understand and fast to train
They give us a baseline


---
# Outline

* Linear regression
 - plot 2D
 - error
 - 3-dim and n-dim ?
* Logistic regression
 - (classification vs regression)
 - plot 2D
* multi class
* linear separability
* L1 and L2 penalty
* (Learning curves and validation curves)
---
# Linear regression

Here we have to predict the value of the target **y** given some observation (explanatory variable) **X**. For illustration purpose, we will consider that the observation is only one dimensional.

.shift-left.pull-right[<img src="../figures/linear_data.svg" width="110%">]

???
Here the target value is expected to be a linear combination of the features
 
---
# Linear regression

We look for the best fit, i.e. we learn w_0 and w_1 such that 
*| y_i - w_o + w_1 * x_i|^2* is minimal for all *i*


.shift-left.pull-left[<img src="../figures/linear_fit.svg" width="100%">]

???

---
# Error in linear regression

For each sample x_i, we than have an error which correspond to |y_i - ŷ_i|^2
(where ŷ_i = w_o + w_1 * x_i)
That correspond to the red bar in the figure below
.shift-left.pull-left[<img src="../figures/linear_fit_red.svg" width="100%">]

???
the fit is the line which minimize the sum of the square of the red lines.

---
# Linear regerssion in higher dimension

If **X** has two dimensions, we obtain a plot like that:

---
# Linear regression with regularization

We could add in the error function the weigths of the parameters.
L1 (Lasso), sparse assumption
L2 (Ridge)

---
# Linear regression with regularization

Left: severals fit on 2 points 
Right: severals fit on 2 points with L2 penalty

.shift-left.pull-left[<img src="../figures/lin_reg_2_points_no_penalty.svg" width="110%">]
.shift-right.pull-right[<img src="../figures/lin_reg_2_points_ridge.svg" width="110%">]

???
http://scipy-lectures.org/packages/scikit-learn/index.html#bias-variance-trade-off-illustration-on-a-simple-regression-problem

Left: As we can see, our linear model captures and amplifies the noise in the data. It displays a lot of variance.

Right: Ridge estimator regularizes the coefficients by shrinking them to zero

As we can see, the estimator displays much less variance. However it systematically under-estimates the coefficient. It displays a biased behavior.

This is a typical example of bias/variance tradeof: non-regularized estimator are not biased, but they can display a lot of variance. Highly-regularized models have little variance, but high bias. This bias is not necessarily a bad thing: what matters is choosing the tradeoff between bias and variance that leads to the best prediction performance. For a specific dataset there is a sweet spot corresponding to the highest complexity that the data can support, depending on the amount of noise and of observations available.

---
# Logistic regression

With Logistic regression, we learn a linear model for classification (and not regresion).
So **y** is either +1 or -1

.shift-left.pull-left[<img src="../figures/categorical.svg" width="110%">]

???
Exemple: succes to an exam after x hours of study.

---
# Logistic regression

sigmoïd: sigma(x) = 1 / (1 + exp(-x))

.shift-left.pull-left[<img src="../figures/logistic_color.svg" width="110%">]


---
# Take home message

* Linear model are good baselines for:
 - regression: linear regression + L2 penalty = Ridge
 - classification: logistic regression

* Better when *p* > *n*


???

.shift-left.pull-left[<img src="../figures/linear_ols.svg" width="110%">]

.pull-right[<img src="../figures/linear_splines.svg" width="110%">]

.shift-left.pull-left[<img src="../figures/linear_ols_test.svg" width="110%">]
.pull-right[<img src="../figures/linear_splines_test.svg" width="110%">]
.centered.reversed[**On new data**]

.shift-left.pull-left[<img src="../figures/ols_simple_test.svg" width="110%">]
.pull-right[<img src="../figures/splines_cubic_test.svg" width="110%">]
.centered[A harder example]
