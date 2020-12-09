class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Overfitting and underfitting

Understand when and why a model generalizes well or not on unseen data.

<img src="../figures/scikit-learn-logo.svg">

???

This lesson covers overfit and underfit, important concepts to understand
why a model generalizes well or not to new data


---
# Which data fit do you prefer?

.shift-left.pull-left[<img src="../figures/linear_ols.svg" width="110%">]

.pull-right[<img src="../figures/linear_splines.svg" width="110%">]


???

Let me start with a simple question: given the following data, which of
the two models do you prefer? I'll give you a moment to think about it.

Most people reply that they prefer the one with the straight line.
However, the one with the wiggly line fits the data perfectly, doesn't it?
So why might we prefer the straight line?

---
# Which data fit do you prefer?

.shift-left.pull-left[<img src="../figures/linear_ols_test.svg" width="110%">]
.pull-right[<img src="../figures/linear_splines_test.svg" width="110%">]
.centered.reversed[**On new data**]

???

Answering this question might be hard. However, in the context of machine
learning we aim for models that generalize. Hence, the good way to frame
the question is: how will the model perform on new data?

---
# Which data fit do you prefer?

.shift-left.pull-left[<img src="../figures/ols_simple_test.svg" width="110%">]
.pull-right[<img src="../figures/splines_cubic_test.svg" width="110%">]
.centered[A harder example]

???

How about a slightly harder example? Which one should we choose?
This is a difficult question.

In this lesson, we will study useful concepts to understand these
tradeoffs.


---
# Varying model complexity

.polynomial[<img src="../figures/polynomial_overfit_0.svg" width=100%>]

* Data generated with 9th-degree polynomial + noise

???

In the latest example, we have generated the data so that y is a
9th-degree polynomial function of X, with some additional noise.


---
# Varying model complexity

.polynomial[<img src="../figures/polynomial_overfit_1.svg" width=100%>]

* Data generated with 9th-degree polynomial + noise

* Fit polynomials of various degrees

???

What we will now do is fit to this data polymonials of various degrees.
We'll start with a polynomial of degree 1: a simple linear regression of
y on X. Clearly, this model does not explain well the data.

---
# Varying model complexity

.polynomial[<img src="../figures/polynomial_overfit_2.svg" width=100%>]

* Data generated with 9th-degree polynomial + noise

* Fit polynomials of various degrees

???

If we fit a polynomial of degree 2, the fit is better.

---
# Varying model complexity

.polynomial[<img src="../figures/polynomial_overfit_5.svg" width=100%>]

* Data generated with 9th-degree polynomial + noise

* Fit polynomials of various degrees

???

Polynomial of degree 5: it's hard to tell whether it explains the data
better or not

---
# Varying model complexity

.polynomial[<img src="../figures/polynomial_overfit_9.svg" width=100%>]

* Data generated with 9th-degree polynomial + noise

* Fit polynomials of various degrees

???

And if we go all the way up to degree 9, the fit actually looks really
bad

---
# Varying model complexity

.polynomial[<img src="../figures/polynomial_overfit.svg" width=100%>]

* Data generated with 9th-degree polynomial + noise

* Fit polynomials of various degrees

???

The actual function that was used to generate the data looks like this,
though we added observational noise.

---
# Overfit: model too complex

.pull-left.shift-left[<img src="../figures/polynomial_overfit_simple_legend.svg" width="110%">]

.pull-right.width50.shift-left[.shift-left[Model too complex for the data:]

* Its best possible fit would approximate well the generative process

* But its flexibility captures noise
]

???

In the case of the polynomial of degree 9, the problem that we face is
that the model that we use is too complex for the data at hand. This
problem is know as overfit in machine learning. With such a rich model,
its best possible fit would approximate well the data-generating process.
Indeed, here we are fitting a polynomial of degree 9 on data generated
with a polynomial of degree 9. However, due to limited data,
the model fit captures noise because it is too flexible.

--

.reversed[**Not enough data** &nbsp; &nbsp; **Too much noise**]

???

This problem is typically encountered when there is not enough data, or too
much noise.

---
# Underfit: model too simple

.pull-left.shift-left[<img src="../figures/polynomial_overfit_assymptotic.svg" width="110%">]

.pull-right.width50.shift-left[Model too simple for the data:

* Its best fit does not approximate well the generative process

* Yet it captures little noise
]

???

At the opposite end of the spectrum, when we are fitting a polynomial of
degree 1, the model is too simple for the data at hand. We say that it
underfits. Its best possible fit cannot approximate well the
data-generating process. On the positive side, it captures little noise,
As a consequence even with limited data, the empirical fit is close to
the best possible fit on an infinite amount of data.

--

.reversed[**Plenty of data** &nbsp; &nbsp; **Low noise**]

???

Underfit is more common when there is plenty of data compared to the
complexity of the model, or in low-noise situations.

---
# Underfit versus overfit

.shift-left.pull-left[<img src="../figures/polynomial_overfit_assymptotic.svg" width="110%">]

.pull-right[<img src="../figures/polynomial_overfit_simple_legend.svg" width="110%">]

.shift-up.pull-left.shift-left[.centered.reversed[Bias]]
.shift-up.pull-right.width50[.centered.reversed[Variance]]

???

So we have these two opposit behaviors:
* underfit, with systematic bias
* and overfit, with large variance

The challenge is to find the right tradeoff between the two.

---
# Underfit versus overfit

.pull-left.width50[<img src="../figures/target_bias.svg" width="80%">]
.pull-right.width50.shift-left[<img src="../figures/target_variance.svg"
				width="80%">]

.shift-up.pull-left.shift-left[.centered.reversed[Bias]]
.shift-up.pull-right.width50[.centered.reversed[Variance]]

???

This bias-variance tradeoff is classic in statistics. Often, adding a
little bit of bias helps reducing the variance. For instance, as with
throwing darts at a target, where throwing the darts less strong might
lead to being below the target on average, but with less scatter.

---

.center[
# Take home messages
]

Models too complex for the data **overfit**:

.tight[
- they explain too well the data that they have seen
- they do not generalize
]

Models too simple for the data **underfit**:

.tight[
- they capture no noise
- but are limited by their expressivity
]

???

When the models are too complex for the data at hand, they overfit. This
means that they explain too well the data that they have seen as they
capture noise, and thus do not generalize to new data.

On the opposite, when models are too simple for the data at hand, they
underfit. This means that they capture no noise, but they not capture all the
structured variations of the data, either: their ability to generalize is then
limited by their expressivity.
