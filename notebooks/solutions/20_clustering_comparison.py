from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

X, y = make_circles(n_samples=1500, 
                    factor=.4, 
                    noise=.05)

km = KMeans(n_clusters=2)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=km.fit_predict(X))

ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=ac.fit_predict(X))

db = DBSCAN(eps=0.2)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=db.fit_predict(X));
