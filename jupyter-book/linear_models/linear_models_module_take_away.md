# Main take-away

## Wrap-up

<!-- Quick wrap-up for the module -->

In this module, we saw that:

- the predictions of a linear model depend on a weighted sum of the values of
  the input features added to an intercept parameter;
- fitting a linear model consists in adjusting both the weight coefficients and
  the intercept to minimize the prediction errors on the training set;
- to train linear models successfully it is often required to scale the input
  features approximately to the same dynamic range;
- regularization can be used to reduce over-fitting: weight coefficients are
  constrained to stay small when fitting;
- the regularization hyperparameter needs to be fine-tuned by cross-validation
  for each new machine learning problem and dataset;
- linear models can be used on problems where the target variable is not
  linearly related to the input features but this requires extra feature
  engineering work to transform the data in order to avoid under-fitting.

## To go further

<!-- Some extra links of content to go further -->

You can refer to the following scikit-learn examples which are related to
the concepts approached during this module:

- [Example of linear regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)
- [Comparison between a linear regression and a ridge regressor](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge_variance.html#sphx-glr-auto-examples-linear-model-plot-ols-ridge-variance-py)
