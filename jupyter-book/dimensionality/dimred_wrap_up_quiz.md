# 🏁 Wrap-up quiz 5

**This quiz requires some programming to be answered.**

Load the `periodic_signals.csv` dataset with the following cell of code. It
contains readings from 170 industrial sensors installed throughout a
manufacturing facility. Each sensor records the average power consumption (in
watts) of a machine, sampled every minute, giving 200 measurements per signal.
Different machines operate with their own characteristic cycles, and a few rare
signals correspond to machinery faults. Here we treat each signal as a point in
a 200-dimensional space and study how **reducing that dimension with PCA**
affects clustering, first with `KMeans` and then with `HDBSCAN`.

```python
import pandas as pd

periodic_signals = pd.read_csv("../datasets/periodic_signals.csv")
_ = periodic_signals.iloc[0].plot(
    xlabel="time (minutes)",
    ylabel="power (Watts)",
    title="Signal from the first sensor",
)
```

Before clustering, let's inspect how much of the information carried by the 200
time points can be summarized by a few principal components. Fit a `PCA` on
the whole dataset (no need to fix the `random_state`) directly on the signals,
**without any scaling**, and look at the cumulative explained variance ratio.

```{admonition} Question
What is the smallest number of principal components needed to retain **at least
90%** of the total variance?

- a) 3
- b) 5
- c) 15
- d) 82

_Select a single answer_

Hint: use
[`numpy.cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
on the `explained_variance_ratio_` attribute of the fitted `PCA`.
```

+++

```{admonition} Question
Now make a pipeline with a `StandardScaler` step before the `PCA` and
recompute the number of components needed to reach 90% of the variance. Select
the true statements:

- a) With `StandardScaler`, more components (around 15) are needed to reach 90%,
  because standardizing gives equal weight to low-variance time points that are
  dominated by noise.
- b) With `StandardScaler`, fewer components are needed, because removing scale
  differences always concentrates the variance.
- c) With `StandardScaler`, the same number of components are needed, because
  scaling doesn't affect the explained variance ratio of PCA.
- d) All measurements are the same physical quantity (power in watts) already
  on a comparable scale, so per-feature standardization is not required here
  and mostly amplifies noise.

_Select all answers that apply_
```

+++

For the rest of the quiz we use PCA **without scaling** and keep
`n_components=5` (the dimension retaining about 90% of the variance). Build a
pipeline made of this `PCA` followed by `KMeans(n_init=3)`, and tune
`n_clusters` with the silhouette score. As in the clustering module, assess
stability by resampling 90% of the data with `train_test_split` for about 20
different `random_state` values, computing the silhouette for `n_clusters` in
`range(2, 11)` each time.

```{admonition} Question
Using the silhouette score after reducing to 5 components, select the true
statements:

- a) The silhouette score is maximized at `n_clusters=7`, which is a very stable
  choice across resamplings.
- b) The silhouette values reach about 0.9, hinting at a strong, well-separated
  cluster structure once PCA has denoised the signals.
- c) The silhouette values stay negative, denoting a bad clustering model.
- d) The best `n_clusters` jumps erratically between 2 and 10, with no stable
  choice.

_Select all answers that apply_
```

+++

```{admonition} Question
Repeat the silhouette analysis for `n_components` equal to 2, 5 and 50 (still
without scaling). Select the true statements about how n_components affects the
silhouette score:

- a) The silhouette score decreases as n_components increases, because in high
     dimension the **relative gap** between intra-cluster and inter-cluster
     distances shrinks.
- b) The silhouette score stays on the same scale across different values of
     n_components, so one can simply maximize it over a grid of
     (n_clusters, n_components) combinations to find the best model.
- c) The silhouette score increases as n_components increases, because in high
     dimension the **relative gap** between intra-cluster and inter-cluster
     distances grows.

_Select a single answer_

```

+++

We now switch to a density-based approach. Create an `HDBSCAN` model with
`min_cluster_size=10` and fit it on the PCA-reduced signals for `n_components`
in `{2, 5, 10, 50}`. Count the number of clusters (excluding the noise label
`-1`) and the number of signals labeled as noise.

```{admonition} Question
Select the true statements:

- a) HDBSCAN finds the same number of clusters regardless of `n_components`.
- b) The number of clusters found by HDBSCAN decreases as `n_components`
  increases, because distance concentration causes separate dense
  regions to merge into fewer, larger detected clusters.
- c) The number of points labeled as noise increases with `n_components`,
  because distance concentration in high dimension makes it harder to
  tell dense regions from sparse ones.
- d) The number of points labeled as noise decreases with `n_components`,
  because points become more spread out and easier to separate into
  distinct dense regions.

_Select all answers that apply_
```

+++

Finally, compare the two approaches. For each PCA dimension, fit
`KMeans(n_clusters=6)` and `HDBSCAN(min_cluster_size=10)` on the reduced data and
compute the Adjusted Mutual Information (AMI) between their labelings.

```{admonition} Question
Select the true statements:

- a) The AMI stays between 0.7 and 0.8 at every `n_components` tested,
  indicating moderate agreement between the two methods.
- b) The AMI stays between 0.8 and 0.9 at every `n_components` tested,
  indicating good agreement between the two methods.
- c) The AMI stays above 0.9 at every `n_components` tested, indicating
  strong agreement between the two methods.

_Select a single answer_
```
