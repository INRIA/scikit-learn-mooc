XX_train = np.concatenate((X_train, np.sin(4 * X_train)), axis=1)
XX_test = np.concatenate((X_test, np.sin(4 * X_test)), axis=1)
regressor.fit(XX_train, y_train)
y_pred_test_sine = regressor.predict(XX_test)

plt.plot(X_test, y_test, 'o', label="data")
plt.plot(X_test, y_pred_test_sine, 'o', label="prediction with sine")
plt.plot(X_test, y_pred_test, label='prediction without sine')
plt.legend(loc='best');
