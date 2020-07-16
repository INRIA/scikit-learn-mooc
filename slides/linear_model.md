class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Linear Model

This lesson covers the linear models.

These are basic models, easy to understand and fast to train.

<img src="../scikit-learn-logo.svg">

???

Linear models are easy to understand and fast to train,
they give us fair baselines.


---
# Outline

* What is a linear model?
* Linear model for regression & classification
* How to avoid overfitting?


---
# An example: Adult census

.very-small[

| Age | Workclass | Education    | Marital-status     | Occupation         | Relationship | Race  | Sex  | Capital-gain | Hours-per-week | Native-country | Salary |
| --- | --------- | ------------ | ------------------ | ------------------ | ------------ | ----- | ---- | ------------ | -------------- | -------------- | ----- |
| 25  | Private   | 11th         | Never-married      | Machine-op-inspct  | Own-child    | Black | Male | 0            | 40             | United-States  | $45k |
| 38  | Private   | HS-grad      | Married-civ-spouse | Farming-fishing    | Husband     | White  | Male | 0            | 50             | United-States   | $40k |
| 28  | Local-gov | Assoc-acdm   | Married-civ-spouse | Protective-serv    | Husband      | White | Male | 0            | 40             | United-States   | $60k  |
]

.shift-left[Salary = *0.4 x* Education + *0.2 x* Hours-per-week + *0.1 x* Age +...]

???
Adult census is here a bit modify, instead of having 2 categories, < $50k and
>= $50k, the target "Salaray" contains the exact value of the salary for each
person. Thus, the target is continuous and we deal with a regression problem
instead of a classification problem.

Salary could be a linear combination of the feature (explanatory variable).



---
# Linear regression

Predict the value of the target **y**  
given some observation **X**

.shift-down.pull-left.shift-left[<img src="../figures/linear_data.svg" width="100%">]

???
For illustration purpose, let's consider 1-dimensional observation,
e.g. salary should be explained by education level (number of year of study)

---
# Linear regression
A linear model is a ramp "as close as possible" to all samples.
The blue curve shows the predictions for each possible **x**

.shift-down.pull-left.shift-left[<img src="../figures/linear_fit.svg" width="100%">]

```python
  from sklearn.linear_model import LinearRegression
  linear_regression = LinearRegression()
  linear_regression.fit(x, y)
```

???

We learn a linear function to predict *y*. Here, the salary is a constant
multiplied by the number of years of study.



---
# Linear regression

The slope is chosen to minimize the distance between the prediction and the
data points

.shift-down.pull-left.shift-left[<img src="../figures/linear_fit_red.svg" width="100%">]


```python
  from sklearn.linear_model import LinearRegression
  linear_regression = LinearRegression()
  linear_regression.fit(x, y)

  y_pred = linear_regression.predict(X)
  residuals = y - y_pred
  error = sum(residuals ** 2)
```

???
An error for each sample corresponds to the red bar in the figure.
The best fit is the blue line which minimizes the sum of (the square of) those
red lines.

Fortunately, there is a formula, given **X** and **y**, to find the optimal
weights in an efficient manner.

---
# Linear regression with several variables

.pull-left.shift-left[<img src="../figures/lin_reg_3D.svg" width="130%">]

The mental picture needs to be extended to several dimensions.

???
With more variables, the mental picture needs to be extended to several
dimensions. However, the idea is the same: a linear model tries to
minimize the error between the predictions, that fall on a plane, and the
data points.

Often, the data have many features, and thus many dimensions. It is not
possible any longer to represent the fitting graphically.

---
# For classification: logistic regression

For **classification**, we use a logistic regression model  
**y** is binary,
either +1 or -1

.shift-left.pull-left[<img src="../figures/categorical.svg" width="100%">]


 ```python
 From sklearn.linear_model import LogisticRegression
 log_reg = LogisticRegression()
 ```

???
FIXME title might appear in two lines on some browser.

Logistic regression is a linear model for **classification** - not regression
as the name would wrongly suggest.

In our `adult_census` dataset, we do not have continuous value for salary but
only whether the salary is higher than $50K or not. This problem is, therefore,
a classification problem.


---
# For classification: logistic regression

The output is now modelled as a form of a step function, which is adjusted on
the data

.shift-left.pull-left[<img src="../figures/logistic_color.svg" width="100%">]


 ```python
 From sklearn.linear_model import LogisticRegression
 log_reg = LogisticRegression()
 log_reg.fit(X, y)
 ```

---
# Logistic regression in 2 dimensions

**X** is 2-dimensional
**y** is the color

.shift-left.pull-left[<img src="../figures/logistic_2D.svg" width="100%">]
.shift-up.shift-left-more.pull-right[
.shift-left-more[<img src="../figures/logistic_3D.svg" width="110%">]
]
 

???
Here is an other way of representing our data.
In this case, X has two dimension x1 and x2.
The axes correspond to x1, x2 and the color corresponds to the target label y.

---
# Model complexity


* A linear model can also overfit


&nbsp;Salary = *0.4 x* Education + *0.2 x* Hours-per-week + *0.1 x* Age
.red[&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; + *0.2 x* Zodiac_sign + *0.3 x* Wear_red_socks + ...]

.small[]

**Regularization** is needed to control model complexity.
The most comon way is to push coefficients toward
small values. Such model is called *Ridge*.

.pull-left[
 ```python
 From sklearn.linear_model import Ridge
 ridge = Ridge(alpha=1)
 ```
]

???

If we have too many parameters with regards to the number of samples, the
linear model can overfit: it assigns non-zero weights to associations by
chance.

The solution is to regularize the model: to foster less complex
solutions. For this, the regularized model is slightly biased to
choose smaller weights for almost a similar fit.

A complexity parameter allows to control the amount of regularization. It
is denominated \alpha. The larger the value of \alpha, the greater the
bias and thus the smaller the coefficients.

---
# Bias-variance tradeoff in Ridge


.pull-left.shift-left[<img src="../figures/lin_reg_2_points.svg" width="110%">]

---
# Bias-variance tradeoff in Ridge


.pull-left.shift-left[<img src="../figures/lin_reg_2_points_no_penalty.svg" width="110%">]
.pull-right[<img src="../figures/lin_reg_2_points_ridge.svg" width="110%">]

.pull-left.shift-left[&nbsp; &nbsp; &nbsp; Low bias, high variance]
.pull-right[&nbsp; &nbsp; &nbsp; High bias, low variance]
???
from http://scipy-lectures.org/packages/scikit-learn/index.html#bias-variance-trade-off-illustration-on-a-simple-regression-problem

Left: As we can see, our linear model captures and amplifies the noise in the
data. It displays a lot of *variance*.

Right: Ridge estimator regularizes the coefficients by shrinking lightly
them to zero.

Ridge displays much less variance. However, it systematically under-estimates
the coefficient. It displays a *biased* behavior.

This is a typical example of bias/variance tradeoff: non-regularized estimator
are not biased, but they can display a lot of variance. Highly-regularized
models have little variance, but high bias. This bias is not necessarily a bad
thing: what matters is choosing the tradeoff between bias and variance that
leads to the best prediction performance. For a specific dataset there is a
sweet spot corresponding to the highest complexity that the data can support,
depending on the amount of noise and of observations available.

---
# Bias-variance tradeoff in Ridge


<img src="../figures/lin_reg_2_points_no_penalty_grey.svg" width="32%">
<img src="../figures/lin_reg_2_points_best_ridge_grey.svg" width="32%">
<img src="../figures/lin_reg_2_points_ridge_grey.svg" width="32%">

.shift-up-less[
Too much variance &nbsp; &nbsp; &nbsp; &nbsp; Best tradeoff &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Too much bias
]

.shift-up-less[
&nbsp; &nbsp; Small alpha &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
Large alpha
]

.width65.shift-up-less.centered[
 ```python
 From sklearn.linear_model import LogisticRegressionCV
 ```
]


???


---
# Regularization in logistic regression

.small[The parameter *C* controls the complexity of the model, and in practice, whether the model focuses on data close to the boundary.]

.shift-up-less.shift-left.pull-left[<img src="../figures/logistic_2D_C0.001.svg" width="90%">]
.shift-up-less.pull-right[<img src="../figures/logistic_2D_C1.svg" width="90%">]
.shift-up.pull-left.shift-left[&nbsp;&nbsp;Small C]
.shift-up.pull-right[&nbsp;&nbsp;Large C]

???
For a large value of C, the model puts more emphasis on the frontier's point.
On the contrary, for a low value of C, the model is considering all the points.
The choice of C depends on the dataset and should be tuned for each set.

---
# Logistic regression for multiclass

Logistic regression can be adapted to **y** containing multiple classes.
There are several options:

.shift-left.pull-left[<img src="../figures/multinomial.svg" width="100%">]
.pull-right[
* Multinomial
* One vs One
* One vs Rest
]
???
Multinomial logistic regression is a natural extension of logistic regression.
Otherwise, we still can run One vs Rest approach.

---
# Linear models may not work on all data


.shift-left.pull-left[<img src="../figures/lin_separable.svg" width="100%">]
.pull-right[<img src="../figures/lin_not_separable.svg" width="100%">]

.pull-left.shift-left[Linearly separable]
.pull-right[*Not* linearly separable]

???
Linear models work as long as your data could be linearly separable.
Otherwise, either we could do feature augmentation (as we will see in an other lesson), or choose a non-linear model.

---
.center[
# Take home messages: Linear models
]

* Good and understandable baselines for:
 - regression: linear regression + regularization = Ridge
 - classification: logistic regression + fine tune `C`

* Very fast to train

* Better when *nb of features* > *nb of samples*


???
