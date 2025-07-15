# 🏁 Wrap-up quiz 4

**This quiz requires some programming to be answered.**

Load the `periodic_signals.csv` dataset with the following cell of code. It
contains readings from 170 industrial sensors installed throughout a
manufacturing facility. Each sensor records the average power consumption (in
watts) every minute for a specific machine, with measurements taken every
minute. Different machines operate with their own characteristic cycles. Rare
events, such as machinery faults or unexpected disturbances, appear as signals
with abnormal frequency patterns. The goal is to identify those disturbances
using the tools we have learned during this module.

```python
import pandas as pd

periodic_signals = pd.read_csv("../datasets/periodic_signals.csv")
_ = periodic_signals.iloc[0].plot(
    xlabel="time (minutes)",
    ylabel="power (Watts)",
    title="Signal from the first sensor",
)
```

Let's see if we can find one or more stable candidates for the number of
clusters (`n_clusters`) using the silhouette score when resampling the
dataset. For such purpose:
- Create a pipeline consisting of a `RobustScaler` (as it is a good scaling
  option when dealing with outliers), followed by `KMeans` with `n_init=5`.
- You can choose to set the `random_state=0` value of the `KMeans` step, but
  fixing it or not should not change the conclusions.
- Generate randomly resampled data consisting of 90% of the data by using
  `train_test_split` with `train_size=0.9`. Change the `random_state` in the
  `train_test_split` to try around 20 different resamplings. You can use the
  `plot_n_clusters_scores` function (or a simplified version of it) inside a
  `for` loop as we did in a previous exercise.
- In each resampling, compute the silhouette score for `n_clusters` varying in
  `range(2, 11)`.

```{admonition} Question
Using the silhouette score heuristics, select the correct statements:

- a) 3 or 4 clusters maximize the score and are resonably stable choices.
- b) 5 or 6 clusters maximize the score and are resonably stable choices.
- c) 7 or 8 clusters maximize the score and are resonably stable choices.
- d) Scores in this range of `n_clusters` are always negative, denoting a bad
  clustering model.
- e) Scores in this range of `n_clusters` are always positive, but hint to a
  weak to moderate cluster cohesion/separation.

_Select all answers that apply_
```

+++

```{admonition} Question
Set `n_clusters=8` in the `KMeans` step of your previous pipeline for the rest
of this quiz. We are going to define an `outlier_score` using the **minimum**
distance to **any** centroid (using the `fit_transform` method of the
pipeline).

What are the indices of the 5 signals that are the farthest from any centroid?

- a) [ 77 32 112 105 101]
- b) [ 92 49 101 132 146]
- c) [ 80 49 121 150 101]
- d) [ 64 98 118 163 121]

_Select a single answer_

Hint: You can make use of
[`numpy.min`](https://numpy.org/doc/stable/reference/generated/numpy.min.html)
and
[`numpy.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html).
Also, remember that the output of `fit_transform` is a numpy array of shape
`(n_samples, n_clusters)`.
```

+++

```{admonition} Question
Create an `HDBSCAN` model (no need for scaling) with `min_cluster_size=10`.
How many clusters (excluding the noise label, which is not a cluster) are
found by this model?

- a) 5
- b) 6
- c) 7
- d) 8

_Select a single answer_
```

+++

How many signals are identified as noise?

- a) 3
- b) 5
- c) 7
- d) 9

_Select a single answer_
```

+++ {"tags": ["solution"]}

solution: c)

The code to count them is the following:

```python
hdbscan_noise_indices = np.where(hdbscan_labels == -1)[0]
n_noise = len(hdbscan_noise_indices)
print(f"{n_noise} signals are labeled as noise.")
```

+++

A priori we don't know if the signals are isotropic or follow a gaussian
distribution in the feature space (i.e. if they form spherical blobs). Because
of that, we don't know if a centroid-based or a density-based clustering is
more suitable. We would like to compare the results from both models, but we
know that the presence of outliers makes the silhouette score tricky to
interpret. We can still use other metrics, such as Adjusted Mutual Information
(AMI), to compare both models.

But first we need k-means to have a similar behavior to HDBSCAN. For such
purpose, we can identify the points that are too far from any centroid as
outliers using the `outlier_score` as defined before. Instead of setting a
fixed distance threshold, we can flag the `n_outliers` signals with the
highest outlier scores as `-1`.

For such purpose:

- Cluster your signals with `KMeans` (using `fit_predict`) to get `kmeans_labels`.
- For a range of values of `n_outliers`, re-label the `n_outliers` with highest
  `outlier_score` to `-1`.
- Compute and plot the AMI between this modified KMeans labeling and the
  HDBSCAN cluster labels as a function of `n_outliers`.

```{admonition} Question
If we denote by `n_noise` the number of signals identified as noise by
HDBSCAN, select the true statements:

- a) AMI reaches a maximum when `n_outliers` < `n_noise`, some points marked
  as noise by HDBSCAN are not clearly isolated from a centroid.
- b) AMI reaches a maximum when `n_outliers` = `n_noise`, the two models
  most strongly agree.
- c) AMI reaches a maximum when `n_outliers` > `n_noise`, k-means has created
  small clusters (with fewer than `min_cluster_size` samples) that match what
  HDBSCAN considers noise.
- d) AMI is too close to zero, indicating coincidences between models
  are mostly random.

_Select a single answer_
```
