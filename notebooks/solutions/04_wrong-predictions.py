for i in incorrect_idx:
    print('%d: Predicted %d True label %d' % (i, pred_y[i], test_y[i]))

# Plot two dimensions

colors = ["darkblue", "darkgreen", "gray"]

for n, color in enumerate(colors):
    idx = np.where(test_y == n)[0]
    plt.scatter(test_X[idx, 0], test_X[idx, 1],
                color=color, label="Class %s" % str(n))

for i, marker in zip(incorrect_idx, ['x', 's', 'v']):
    plt.scatter(test_X[i, 0], test_X[i, 1],
                color="darkred",
                marker=marker,
                s=40,
                label=i)

# Make xlim larger to accommodate legend
plt.xlim(0, 9)
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
plt.legend(loc=3)
plt.title("Iris Classification results")
plt.show()
