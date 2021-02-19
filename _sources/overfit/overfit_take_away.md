# Main take-away

## Wrap-up

- **Overfitting** is caused by the **limited size of the training set**, the
  **noise** in the data, and the **high flexibility** of common machine learning
  models.

- **Underfitting** happens when the learnt prediction functions suffer from
  **systematic errors**. This can be caused by a choice of model family and
  parameters, which leads to a **lack of flexibility** to capture the repeatable
  structure of the true data generating process.

- For a fixed training set, the objective is to **minimize the test error** by
  adjusting the model family and its parameters to find the
  **best trade-off between overfitting for underfitting**.

- For a given choice of model family and parameters, **increasing the
  training set size will decrease overfitting** but can also cause an increase
  of underfitting.

- The test error of a model that is neither overfitting nor underfitting can
  still be high if the variations of the target variable cannot be fully
  determined by the input features. This irreducible error is caused by what we
  sometimes call label noise. In practice, this often happens when we do not
  have access to important features for one reason or another.

## To go further

It is possible to give a precise mathematical treatment of the bias and the
variance of a regression model. The Wikipedia article on the [Bias-variance
tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) explains
how the **squared test error can be decomposed as the sum of the squared bias,
the variance and the irreducible error** for a given regression error.

The next chapters on linear models, decision trees and ensembles will give
concrete examples on how to diagnose and how to tackle overfitting and
underfitting.

You can refer to the following scikit-learn examples which are related to
the concepts approached during this module:

- [Illustration of underfitting and overfitting concepts](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py)
- [Difference between train and test scores](https://scikit-learn.org/stable/auto_examples/model_selection/plot_train_error_vs_test_error.html#sphx-glr-auto-examples-model-selection-plot-train-error-vs-test-error-py)
- [Example of a validation curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)
