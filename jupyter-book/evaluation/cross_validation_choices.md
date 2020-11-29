# Choice of cross-validation

In the previous section, we presented the cross-validation framework.
However, we always use the `ShuffleSplit` strategy to repeat the split. One
should question if this approach is always the best option and that some
other cross-validation strategies would be better adapted. Indeed, we will
focus on three aspects that influenced the choice of the cross-validation
strategy: class stratification, sample grouping, feature dependence.

```{tableofcontents}

```
