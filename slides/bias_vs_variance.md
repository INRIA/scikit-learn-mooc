class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Bias and Variance

A statistical view of Underfitting and Overfitting.

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



.center[
# Take home messages
]

**High bias** == **underfitting**:

.tight[
- systematic prediction errors
- the model prefers to ignore some aspects of the data
]

**High variance** == **overfitting**:

.tight[
- prediction errors without obvious structure
- small change in the training set large change in model
]

The bias can come from the choice of the model family.


