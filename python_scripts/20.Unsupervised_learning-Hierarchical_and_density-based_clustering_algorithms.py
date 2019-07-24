# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "2de13356-e9ae-466c-89d1-50618945c658"}}
# %matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "4a9d75ee-def8-451e-836f-707a63d8ea90"}}
# # Unsupervised learning: Hierarchical and density-based clustering algorithms

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "2e676319-4de0-4ee0-84ec-f525353b5195"}}
# In a previous notebook, "08 Unsupervised Learning - Clustering.ipynb", we introduced one of the essential and widely used clustering algorithms, K-means. One of the advantages of K-means is that it is extremely easy to implement, and it is also computationally very efficient compared to other clustering algorithms. However, we've seen that one of the weaknesses of K-Means is that it only works well if the data can be grouped into a globular or spherical shape. Also, we have to assign the number of clusters, *k*, *a priori* -- this can be a problem if we have no prior knowledge about how many clusters we expect to find. 

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "7f44eab5-590f-4228-acdb-4fd1d187a441"}}
# In this notebook, we will take a look at 2 alternative approaches to clustering, hierarchical clustering and density-based clustering. 

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "a9b317b4-49cb-47e0-8f69-5f6ad2491370"}}
# # Hierarchical Clustering

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "d70d19aa-a949-4942-89c0-8c4911bbc733"}}
# One nice feature of hierachical clustering is that we can visualize the results as a dendrogram, a hierachical tree. Using the visualization, we can then decide how "deep" we want to cluster the dataset by setting a "depth" threshold. Or in other words, we don't need to make a decision about the number of clusters upfront.
#
# **Agglomerative and divisive hierarchical clustering**
#
# Furthermore, we can distinguish between 2 main approaches to hierarchical clustering: Divisive clustering and agglomerative clustering. In agglomerative clustering, we start with a single sample from our dataset and iteratively merge it with other samples to form clusters -- we can see it as a bottom-up approach for building the clustering dendrogram.  
# In divisive clustering, however, we start with the whole dataset as one cluster, and we iteratively split it into smaller subclusters -- a top-down approach.  
#
# In this notebook, we will use **agglomerative** clustering.

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "d448e9d1-f80d-4bf4-a322-9af800ce359c"}}
# **Single and complete linkage**
#
# Now, the next question is how we measure the similarity between samples. One approach is the familiar Euclidean distance metric that we already used via the K-Means algorithm. As a refresher, the distance between 2 m-dimensional vectors $\mathbf{p}$ and $\mathbf{q}$ can be computed as:
#
# \begin{align} \mathrm{d}(\mathbf{q},\mathbf{p}) & = \sqrt{(q_1-p_1)^2 + (q_2-p_2)^2 + \cdots + (q_m-p_m)^2} \\[8pt]
# & = \sqrt{\sum_{j=1}^m (q_j-p_j)^2}.\end{align}	
#

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "045c17ed-c253-4b84-813b-0f3f2c4eee3a"}}
# However, that's the distance between 2 samples. Now, how do we compute the similarity between subclusters of samples in order to decide which clusters to merge when constructing the dendrogram? I.e., our goal is to iteratively merge the most similar pairs of clusters until only one big cluster remains. There are many different approaches to this, for example single and complete linkage. 
#
# In single linkage, we take the pair of the most similar samples (based on the Euclidean distance, for example) in each cluster, and merge the two clusters which have the most similar 2 members into one new, bigger cluster.
#
# In complete linkage, we compare the pairs of the two most dissimilar members of each cluster with each other, and we merge the 2 clusters where the distance between its 2 most dissimilar members is smallest.
#
# ![](figures/clustering-linkage.png)
#

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "b6cc173c-044c-4a59-8a51-ec81eb2a1098"}}
# To see the agglomerative, hierarchical clustering approach in action, let us load the familiar Iris dataset -- pretending we don't know the true class labels and want to find out how many different follow species it consists of:

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "b552a94c-9dc1-4c76-9d9b-90a47cd7811a"}}
from sklearn.datasets import load_iris
from figures import cm3

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
n_samples, n_features = X.shape

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm3)

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "473764d4-3610-43e8-94a0-d62731dd5a1c"}}
# First, we start with some exploratory clustering, visualizing the clustering dendrogram using SciPy's `linkage` and `dendrogram` functions:

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "d7f4a0e0-5b4f-4e08-9c77-fd1b1d13c877"}}
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

clusters = linkage(X, 
                   metric='euclidean',
                   method='complete')

dendr = dendrogram(clusters)

plt.ylabel('Euclidean Distance')

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "68cb3270-9d4b-450f-9372-58989fe93a3d"}}
# Next, let's use the `AgglomerativeClustering` estimator from scikit-learn and divide the dataset into 3 clusters. Can you guess which 3 clusters from the dendrogram it will reproduce?

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "4746ea9e-3206-4e5a-bf06-8e2cd49c48d1"}}
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')

prediction = ac.fit_predict(X)
print('Cluster labels: %s\n' % prediction)

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "a4e419ac-a735-442e-96bd-b90e60691f97"}}
plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap=cm3)

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "63c6aeb6-3b8f-40f4-b1a8-b5e2526beaa5"}}
# # Density-based Clustering - DBSCAN

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "688a6a37-3a28-40c8-81ba-f5c92f6d7aa8"}}
# Another useful approach to clustering is *Density-based Spatial Clustering of Applications with Noise* (DBSCAN). In essence, we can think of DBSCAN as an algorithm that divides the dataset into subgroup based on dense regions of point.
#
# In DBSCAN, we distinguish between 3 different "points":
#
# - Core points: A core point is a point that has at least a minimum number of other points (MinPts) in its radius epsilon.
# - Border points: A border point is a point that is not a core point, since it doesn't have enough MinPts in its neighborhood, but lies within the radius epsilon of a core point.
# - Noise points: All other points that are neither core points nor border points.
#
# ![](figures/dbscan.png)
#
# A nice feature about DBSCAN is that we don't have to specify a number of clusters upfront. However, it requires the setting of additional hyperparameters such as the value for MinPts and the radius epsilon.

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "98acb13b-bbf6-412e-a7eb-cc096c34dca1"}}
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=400,
                  noise=0.1,
                  random_state=1)
plt.scatter(X[:,0], X[:,1])
plt.show()

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "86c183f7-0889-443c-b989-219a2c9a1aad"}}
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2,
            min_samples=10,
            metric='euclidean')
prediction = db.fit_predict(X)

print("Predicted labels:\n", prediction)

plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap=cm3)

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "84c2fb5c-a984-4a8e-baff-0eee2cbf0184"}}
# # Exercise

# %% [markdown] {"deletable": true, "editable": true, "nbpresent": {"id": "6881939d-0bfe-4768-9342-1fc68a0b8dbc"}}
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Using the following toy dataset, two concentric circles, experiment with the three different clustering algorithms that we used so far: `KMeans`, `AgglomerativeClustering`, and `DBSCAN`.
#
# Which clustering algorithms reproduces or discovers the hidden structure (pretending we don't know `y`) best?
#
# Can you explain why this particular algorithm is a good choice while the other 2 "fail"?
#       </li>
#     </ul>
# </div>

# %% {"deletable": true, "editable": true, "nbpresent": {"id": "4ad922fc-9e38-4d1d-b0ed-b0654c1c483a"}}
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1500, 
                    factor=.4, 
                    noise=.05)

plt.scatter(X[:, 0], X[:, 1], c=y);

# %% {"deletable": true, "editable": true}
# # %load solutions/20_clustering_comparison.py
