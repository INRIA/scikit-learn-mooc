class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Bias and Variance

A statistical view of Underfitting and Overfitting.

---
# Resampling the training set

- Limited amount of training data

- Training set is taken at random

- What is the impact of this choice of training set on the learned prediction
  function?


???
Machine learning operates with finite training set:

We label an arbitrarily random subset of all possible observations because
labeling all the possible observations would be too costly.

What if we used a different training set?

- How different would be the resulting learned prediction functions?

- What would be their average test error?


---
# Underfit: bias

.pull-left.shift-left[<img src="../figures/polynomial_overfit_assymptotic.svg" width="110%">]

.pull-right.width50.shift-left[<img src="../figures/target_bias.svg"
				class="shift-up" width="90%">]

???

Underfit leads to systematic biases: 
the predictions cannot be on target on average, because the model that we
use to predict is systematically off the data-generating process.

---
# Overfit: variance 

.pull-left.shift-left[<img src="../figures/polynomial_overfit_simple_legend.svg" width="110%">]

.pull-right.width50.shift-left[<img src="../figures/target_variance.svg"
				class="shift-up" width="90%">]

???

The problem of overfit is one of variance: on average, the predictions
are not really off, but each tends to fall far from the target. This can
be seen by their large spread around the best possible prediction. A useful
mental picture is that of the spread of arrows on a target.


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

**High bias** == **underfitting**:

.tight[
- systematic prediction errors
- the model prefers to ignore some aspects of the data
- mispecified models
]

**High variance** == **overfitting**:

.tight[
- prediction errors without obvious structure
- small change in the training set, large change in model
- unstable models
]

The bias can come from the choice of the model family.
