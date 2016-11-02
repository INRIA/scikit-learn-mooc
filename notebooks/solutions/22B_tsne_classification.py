from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=1)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
print('KNeighborsClassifier accuracy without t-SNE: {}'.format(clf.score(X_test, y_test)))

tsne = TSNE(random_state=42)
digits_tsne_train = tsne.fit_transform(X_train)
digits_tsne_test = tsne.fit_transform(X_test)

clf = KNeighborsClassifier()
clf.fit(digits_tsne_train, y_train)
print('KNeighborsClassifier accuracy with t-SNE: {}'.format(clf.score(digits_tsne_test, y_test)))
