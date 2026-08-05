# Module overview

## What you will learn

<!-- Give in plain English what the module is about -->

This module gives an intuitive introduction to dimensionality reduction.

We focus on Principal Component Analysis (PCA), the most widely used
dimensionality reduction technique. PCA is simple enough to build geometric
intuitions on, yet rich enough to raise non-trivial questions: how do you
preprocess features before applying it, how many components should you keep, and
what do you do when the standard heuristics break down, as they do for text
data?

The module builds on the supervised pipelines from the Linear Models and
Selecting The Best Model modules, and on the unsupervised foundations from the
Clustering module. We extend those ideas in two directions. First, we treat
dimensionality reduction as a preprocessing step inside supervised and
unsupervised pipelines, and show how to tune it. Second, we apply it to text
data, where the feature space can have thousands of dimensions and the usual
rules of thumb stop working. There we also step outside scikit-learn to compare
PCA against non-linear techniques such as t-SNE and UMAP, which reveal cluster
structure that linear projections compress away.

## Before getting started

<!-- Give the required skills for the module -->

The required technical skills to carry on this module are:


- skills acquired during the "Selecting The Best Model" and "Linear Models"
  modules for basic concepts around hyperparameter stability.

- skills acquired during the "Clustering" module for basic concepts in
  unsupervised learning and for text data preprocessing.

<!-- Point to resources to learning these skills -->

## Objectives and time schedule

<!-- Give the learning objectives -->

The objective in the module are the following:

- Build geometric intuitions on PCA
- Understand why and how to scale features before applying PCA
- Use heatmaps to interpret how original features contribute to each component
- Tune `n_components` as a hyperparameter in a supervised pipeline
- Choose `n_components` in the unsupervised case using explained variance
  curves, the Kaiser criterion, and silhouette scores, and understand when each
  criterion is appropriate
- Understand why standard heuristics for choosing `n_components` break down for
  text data, and what practitioners use instead
- Compare linear (PCA) and non-linear (t-SNE, UMAP) dimensionality reduction
  techniques and understand when each is most informative

<!-- Give the investment in time -->

The estimated time to go through this module is about 3 hours.
