# Main take-away

## Wrap-up

- Overfitting is caused by the limited size of the training set, the noise in
  the data and the high flexibility of common machine learning models.

- Underfitting happens when the learned prediction functions make systematic
  errors. This can be caused by a choice of the model family and parameters that
  lead to a lack of flexibility to capture the repeatable structure of the true
  data generating process.

- For a fixed training set, one strives to minimize the test error by adjusting
  the choice of the model family and the model parameters to find the best
  trade-off between overfitting for underfitting.

- For a fixed choice model family and parameters, increasing the training set
  size will often decrease overfitting but can cause an increase overfitting.

- The most effective way to reduce overfitting is to collect more labeled data
  to increase the training set size. However this is not always easy to do in
  practice.

- The test error of a model that is neither overfitting nor underfitting can
  still be high if the variations of the target variable cannot be fully
  determined by the input features. This irreducible error is caused by what we
  sometimes call label noise. In practice, this often happens when we do not
  have access to important features for one reason or another.

## To go further

It is possible to give a precise mathematical treatment of the bias and the
variance of a regression model. The Wikipedia article on the [Bias-variance
traceoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) explains
how the **squared test error can be decomposed as the sum of the squared bias,
the variance and the irreducible error** for a given regression error.

The next chapters on linear models, decision trees and ensembles will give
concrete examples on how to diagnose and how to tackle overfitting and
underfitting.
