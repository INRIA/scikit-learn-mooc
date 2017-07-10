
nu = 0.05  # theory says it should be an upper bound of the fraction of outliers

for gamma in [0.001, 1.]:
    ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
    ocsvm.fit(X)

    Z_ocsvm = ocsvm.decision_function(grid)
    Z_ocsvm = Z_ocsvm.reshape(xx.shape)

    plt.figure()
    c_0 = plt.contour(xx, yy, Z_ocsvm, levels=[0], colors='red', linewidths=3)
    plt.clabel(c_0, inline=1, fontsize=15, fmt={0: str(alpha_set)})
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], color='red')
    plt.legend()
    plt.show()
