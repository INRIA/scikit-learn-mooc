k = 1  # change to see other numbers

X_k = X[y == k]

iforest = IsolationForest(contamination=0.05)
iforest = iforest.fit(X_k)
iforest_X = iforest.decision_function(X_k)

X_strong_outliers = X_k[np.argsort(iforest_X)[:10]]

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in zip(range(len(X_strong_outliers)), axes.ravel()):
    ax.imshow(X_strong_outliers[i].reshape((8, 8)),
               cmap=plt.cm.gray_r, interpolation='nearest')
    ax.axis('off')
