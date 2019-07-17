for i in incorrect_idx:
    print('%d: Predicted %d True label %d' % (i, pred_y[i], test_y[i]))

# Plot two dimensions

colors = ["darkblue", "darkgreen", "gray"]

for n, color in enumerate(colors):
    idx = np.where(test_y == n)[0]
    plt.scatter(test_X[idx, 1], test_X[idx, 2],
                color=color, label="Class %s" % str(n))

for i, marker in zip(incorrect_idx, ['x', 's', 'v']):
    plt.scatter(test_X[i, 1], test_X[i, 2],
                color="darkred",
                marker=marker,
                s=40,
                label=i)

plt.xlabel('sepal width [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc=1, scatterpoints=1)
plt.title("Iris Classification results")
plt.show()
