from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

plt.figure(figsize=(14, 4))
for i, est in enumerate([PCA(n_components=2, whiten=True),
                         Isomap(n_components=2, n_neighbors=10),
                         LocallyLinearEmbedding(n_components=2, n_neighbors=10, method='modified')]):
    plt.subplot(131 + i)
    projection = est.fit_transform(digits.data)
    plt.scatter(projection[:, 0], projection[:, 1], c=digits.target)
    plt.title(est.__class__.__name__)
