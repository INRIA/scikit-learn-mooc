from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
digits_isomap = iso.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_isomap[:, 0].min(), digits_isomap[:, 0].max() + 1)
plt.ylim(digits_isomap[:, 1].min(), digits_isomap[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_isomap[i, 0], digits_isomap[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
