# Choice of cross-validation

In the previous notebooks, we presented the cross-validation framework.
However, we always used either a default `KFold` or a `ShuffleSplit` strategies
to iteratively split our dataset. However, you should not assume that these
approaches are always the best option: some other cross-validation strategies
might be better adapted to your problem. Indeed, we will focus on three aspects
that influenced the choice of the cross-validation strategy: class
stratification, sample grouping, and feature dependence.

```{tableofcontents}

```
