# Module overview

## What you will learn

<!-- Give in plain English what the module is about -->

In the previous modules, we introduced the development, tuning and evaluation
of **supervised** machine learning models and pipelines.

In this module we present an **unsupervised** learning task, namely clustering.
In particular, we will focus on the k-means algorithm, and consider how to
evaluate such models via concepts such as cluster stability and evaluation
metrics such as silhouette score and inertia. We also introduce supervised
clustering metrics that leverage annotated data to assess clustering
quality.

Finally, we discuss what to do when the assumptions of k-means do not hold, such
as using HDBSCAN for non-convex clusters, and show how k-means can still be
useful as a feature engineering step in a supervised learning pipeline, by using
distances to centroids.


## Before getting started

<!-- Give the required skills for the module -->

The required technical skills to carry on this module are:

- skills acquired during the "The Predictive Modeling Pipeline" module with
  basic usage of scikit-learn;
- skills acquired during the "Selecting The Best Model" module, mainly around
  the concept of validation curves and the concepts around stability.

<!-- Point to resources to learning these skills -->

## Objectives and time schedule

<!-- Give the learning objectives -->

The objective in the module are the following:

- apply k-means clustering and assess its behavior across different settings;
- evaluate cluster quality using unsupervised metrics such as silhouette score
  and WCSS (also known as inertia);
- interpret and compute supervised clustering metrics (e.g., AMI, ARI,
  V-measure) when ground truth labels are available;
- understand the limitations of k-means and identify cases where its assumptions
  (e.g., convex, isotropic clusters) do not hold;
- use HDBSCAN as an alternative clustering method suited for irregular or
  non-convex cluster shapes;
- integrate k-means into a supervised learning pipeline by using distances to
  centroids as features.

<!-- Give the investment in time -->

The estimated time to go through this module is about 6 hours.
