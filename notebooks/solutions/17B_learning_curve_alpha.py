X, y, true_coefficient = make_regression(n_samples=200, n_features=30, n_informative=10, noise=100, coef=True, random_state=5)

plt.figure(figsize=(10, 5))
plt.title('alpha=1')
plot_learning_curve(LinearRegression(), X, y)
plot_learning_curve(Ridge(alpha=1), X, y)
plot_learning_curve(Lasso(alpha=1), X, y)

plt.figure(figsize=(10, 5))
plt.title('alpha=10')
plot_learning_curve(LinearRegression(), X, y)
plot_learning_curve(Ridge(alpha=10), X, y)
plot_learning_curve(Lasso(alpha=10), X, y)

plt.figure(figsize=(10, 5))
plt.title('alpha=100')
plot_learning_curve(LinearRegression(), X, y)
plot_learning_curve(Ridge(alpha=100), X, y)
plot_learning_curve(Lasso(alpha=100), X, y)
