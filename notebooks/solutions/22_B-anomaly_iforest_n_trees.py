for n_estimators in [1, 10, 50, 100]:
    iforest = IsolationForest(n_estimators=n_estimators, contamination=0.10)
    iforest = iforest.fit(X)

    Z_iforest = iforest.decision_function(grid)
    Z_iforest = Z_iforest.reshape(xx.shape)

    plt.figure()
    c_0 = plt.contour(xx, yy, Z_iforest,
                      levels=[iforest.threshold_],
                      colors='red', linewidths=3)
    plt.clabel(c_0, inline=1, fontsize=15,
               fmt={iforest.threshold_: str(alpha_set)})
    plt.scatter(X[:, 0], X[:, 1], s=1.)
    plt.legend()
    plt.show()