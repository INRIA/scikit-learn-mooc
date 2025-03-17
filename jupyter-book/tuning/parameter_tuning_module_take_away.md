# Main take-away

## Wrap-up

<!-- Quick wrap-up for the module -->

- Hyperparameters have an impact on the models' performance and should be
  wisely chosen;
- The search for the best hyperparameters can be automated with a grid-search
  approach or a randomized search approach;
- A grid-search can be computationally expensive and becomes less attractive as
  the number of hyperparameters to explore increases. Moreover, the combinations
  are sampled on a fixed, regular grid.
- A randomized-search allows exploring within a fixed budget, even as the number
  of hyperparameters increases. In this case, combinations can be sampled either
  on a regular grid or from a given distribution.

## To go further

<!-- Some extra links of content to go further -->

You can refer to the following scikit-learn examples which are related to
the concepts approached during this module:

- [Example of a grid-search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py)
- [Example of a randomized-search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py)
- [Example of a nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py)
