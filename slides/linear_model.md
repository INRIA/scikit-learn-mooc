class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Linear Model

This lesson covers the linear models.

These are basic models, easy to understand and fast to train.

<img src="../figures/scikit-learn-logo.svg">

???

Linear models are easy to understand and fast to train.
They are typically good baselines.

We will cover intuitions on how they work in a machine learning 
settings.


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

Let us consider a variant of the adult census data that we saw
previously: instead of having 2 categories, *< $50k* and *>= $50k*, the
target "Salary" contains the exact value of the salary for each person.
Thus, the target is continuous, so we are dealing with a regression problem
instead of a classification problem.

The linear model assumes that the salary can be explained as a linear
combination of the features (explanatory variable), for instance 0.4 x
Education + 0.2 x Hours-per-week + 0.1 x Age.


---
# Linear regression

Predict the value of the target **y**  
given some observation **X**

.shift-down.pull-left.shift-left[<img src="../figures/linear_data.svg" width="100%">]

???

For illustration purpose, let's consider a 1-dimensional observations:
explaining the salary as a function of a single feature, for instance the
education level (the number of years of study).

---
# Linear regression

A linear model is a ramp "as close as possible" to all samples.
The blue curve shows the predictions for any possible **x**

.shift-down.pull-left.shift-left[<img src="../figures/linear_fit.svg" width="100%">]

```python
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x, y)
```

???

We learn a linear function to predict *y*. Here, the salary is expressed
as a constant multiplied by the number of years of study.

Learning this function consists in finding the straight line which is
as close as possible as all the data points. 

The corresponding model can then be used to make predictions for any
possible **x**, as displayed by the blue line.

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
```

???

The slope of the line is chosen to minimize the distance between the
prediction and the data points. This distance constitutes an error
for each sample shown as the red bar on the figure.

The best fit is the blue line which minimizes the sum of the square of
those red lines.

Fortunately, scikit-learn has an estimator, the `LinearRegression`
object, that computes this for us.

---
# Linear regression with several variables

.pull-left.shift-left[<img src="../figures/lin_reg_3D.svg" width="130%">]

The mental picture needs to be extended to several dimensions.

???

With more variables, the mental picture needs to be extended to several
dimensions. However, the idea is the same: a linear model tries to
minimize the error between the predictions and the data points.
The predictions now form a plane.

Often, the data have many features, and thus many dimensions. It is no
longer possible to visualize the fitting with a simple figure.

---
# For classification: logistic regression

For **classification**, we use a logistic regression model

**y** is binary, either +1 or -1

.shift-left.pull-left[<img src="../figures/categorical.svg" width="100%">]


 ```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
 ```

???
FIXME title might appear in two lines on some browser.

In our `adult_census` dataset, we do not have continuous values for salary but
only whether the salary is higher than $50K. This problem is, therefore,
a classification problem.

The prediction target, **y**, is binary. It can be represented by either
+1 or -1. However, a straight line is not suited to try to explain
such binary target.

Hence, dedicated linear models for classification are needed. *Logistic
regression* is such a model: it is intended for **classification** and
not regression as the name would wrongly suggest.


---
# For classification: logistic regression

The output is now modelled as a form of a step function, which is adjusted on
the data

.shift-left.pull-left[<img src="../figures/logistic_color.svg" width="100%">]


 ```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
 ```


???

With logistic regression, the output is modelled using a form of soft
step function, adjusted to the data. This function is called a logistic
function. Using a soft, graduate shift between *y = -1* and *y = +1* is
useful to capture the grey zone, where the value of *x* does not
completely tell us whether the target value is -1 or +1.

In scikit-learn, this is done with the `LogisticRegression` object.

---
# Logistic regression in 2 dimensions

**X** is 2-dimensional
**y** is the color

.shift-up.shift-left.pull-left[<img src="../figures/logistic_3D.svg" width="110%">]
.shift-right-more.pull-right[
    <img src="../figures/logistic_2D.svg" width="100%">
]
 

???

If the data has two features, it is convenient to represent it
differently.

Here, *X* has two dimension *x1* and *x2*.

The figure on the left shows a representation similar to before: the
features now appear as two dimensions, and the target to predict is in
the third dimension. The soft step defining the prediction is now a
surface.

A more synthetic visualization is visible on the figure on the right: the
two axes correspond to *x1* and *x2*. The data points are represented as dots,
with the two different colors corresponding to two different values of
the target label y. The model predictions appear as straight lines,
separating the two cloud of points *y=-1* and *y=+1*, which correspond to
the surface on the left.

This last visualization is commonly used in machine learning.

---
# Model complexity


* A linear model can also overfit


&nbsp;Salary = *0.4 x* Education + *0.2 x* Hours-per-week + *0.1 x* Age
.red[&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; + *0.2 x* Zodiac_sign + *0.3 x* Wear_red_socks + ...]

.small[]

**Regularization** is needed to control model complexity.
The most common way is to push the coefficients toward
small values. Such model is called *Ridge*.

.pull-left[
 ```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
 ```
]

???

If we have too many parameters in regard to the number of samples, the
linear model can overfit: it assigns non-zero weights to associations by
chance.

As described in a previous lecture, the problem with overfit is that the
model learns a decision function that is too complicated: here the
non-zero associations to unrelated factors such as wearing red socks. As
a consequence, the model generalizes poorly.

The solution is to regularize the model: to foster less complex
solutions. For this purpose, a linear model can be regularized by
slightly biasing to choose smaller weights for almost a similar fit.

The `Ridge` estimator does this in scikit-learn.

This model comes with a complexity parameter that controls the amount of
regularization. This parameter is named *alpha*. The larger the value of
*alpha*, the greater the bias, and thus the smaller the coefficients.

---
# Bias-variance tradeoff in Ridge


.pull-left.shift-left[<img src="../figures/lin_reg_2_points.svg" width="110%">]

.pull-left.shift-left[&nbsp; &nbsp; &nbsp; Low bias, high variance]

???

Let's illustrate the ridge's bias-variance tradeoff.

With 2 data points, a non-biased linear model fits perfectly the data.


---
# Bias-variance tradeoff in Ridge


.pull-left.shift-left[<img src="../figures/lin_reg_2_points_no_penalty.svg" width="110%">]
.pull-right[<img src="../figures/lin_reg_2_points_ridge.svg" width="110%">]

.pull-left.shift-left[&nbsp; &nbsp; &nbsp; Low bias, high variance]
.pull-right[&nbsp; &nbsp; &nbsp; High bias, low variance]
???

When there is noise in the data, the non-biased linear model captures
and amplifies this noise. As a result, it displays a lot of *variance* 
in its predictions.

On the right, we have a ridge estimator with a large value of *alpha*,
regularizing the coefficients by shrinking them to zero.

The ridge displays much less variance. However, it systematically
under-estimates the coefficient. It displays a **biased** behavior.

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
from sklearn.linear_model import RidgeCV
 ```
]


???

This is a typical example of bias/variance tradeoff: non-regularized
estimator are not biased, but they can display a lot of variance.
Highly-regularized models have little variance, but high bias.

This bias is not necessarily a bad thing: what matters is choosing the
tradeoff between bias and variance that leads to the best prediction
performance. For a specific dataset there is a sweet spot corresponding
to the highest complexity that the data can support, depending on the
amount of noise and observations available.

Given new data points, beyond our two initial measures, the sweep spot
minimizes the error. For the specific case of the `Ridge` estimator, in
scikit-learn, the best value of *alpha* can be automatically found
using the `RidgeCV` object.

Note that, in general, for prediction, it is always better to prefer
`Ridge` over a `LinearRegression` object. Using at least a small amount
of regularization is always useful.

---
# Regularization in logistic regression

.small[The parameter *C* controls the complexity of the model, and in practice, whether the model focuses on data close to the boundary.]

.shift-up-less.shift-left.pull-left[<img src="../figures/logistic_2D_C0.001.svg" width="90%">]
.shift-up-less.pull-right[<img src="../figures/logistic_2D_C1.svg" width="90%">]
.shift-up.pull-left.shift-left[&nbsp;&nbsp;Small C]
.shift-up.pull-right[&nbsp;&nbsp;Large C]

.width65.shift-up-less.centered[
 ```python
from sklearn.linear_model import LogisticRegressionCV
 ```
]

???

For classification, logistic regression also comes with regularization.


In scikit-learn, this regularization is controlled by a parameter called
*C*, which has a slightly different behavior than *alpha* in the Ridge
estimator.

For a large value of *C*, the model puts more emphasis on the data points
close to the frontier.
On the contrary, for a low value of *C*, the model considers all the points.

As with Ridge, the tradeoff controlled by the choice of *C* depends on
the dataset and should be tuned for each set. This tuning can be done in
scikit-learn using the `LogisticRegressionCV` object.


---
# Logistic regression for multiclass

Logistic regression can be adapted to **y** containing multiple classes.
There are several options:

.shift-left.pull-left[<img src="../figures/multinomial.svg" width="100%">]
.pull-right[
* Multinomial
* One versus One
* One versus Rest
]
???

So far, we have considered the case where the output **y** is binary.
When there is more than 2 classes to choose from, more than one decision
boundary is needed.

The `LogisticRegression` estimator has strategies to deal transparently
with such settings, known as multiclass settings.

For instance, the "multinomial" is a natural extension of the logistic,
using a function with several soft steps. There are also **One versus One**
and **One versus Rest** approaches that learn decisions discriminating 
either one class versus every individual class, or one class versus all the 
other classes.

---
# Linear models are not suited to all data


.shift-left.pull-left[<img src="../figures/lin_separable.svg" width="100%">]
.pull-right[<img src="../figures/lin_not_separable.svg" width="100%">]

.pull-left.shift-left[Linearly separable]
.pull-right[*Not* linearly separable]

???

Linear models work well if the classes are linearly separable.

However, sometimes, the best decision boundary to separate classes is not 
well approximated by a line.

In such a situation, we can either use non-linear models, or perform
transformations on the data, known as feature augmentation. We will 
cover these in other lessons.

---
.center[
# Take home messages: Linear models
]

* Good and understandable baselines for:
 - regression: linear regression + regularization = Ridge
 - classification: logistic regression

* Fast to train

* Better when *nb of features* > *nb of samples*


???

To summarize on linear models:

They form good baselines that can be easily understood. A later lesson
will cover in details the intuitive interpretations of linear-model
coefficients.

For regression, a good choice is typically to use a Ridge regression,
which adds a simple regularization.

For classification, a good choice is to use a logistic regression

These models are fast to train, and hence facilitate work.

In addition, they are particularly useful when the number of features is
larger than the number of samples.
