# suppress warnings from older versions of KNeighbors
import warnings
warnings.filterwarnings('ignore', message='kneighbors*')

X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)

for Model in [LinearSVC, GaussianNB, KNeighborsClassifier]:
    clf = Model().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print Model.__name__, metrics.f1_score(y_test, y_pred)
    
print '------------------'

# test SVC loss
for loss in ['l1', 'l2']:
    clf = LinearSVC(loss=loss).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "LinearSVC(loss='{0}')".format(loss), metrics.f1_score(y_test, y_pred)
    
print '-------------------'
    
# test K-neighbors
for n_neighbors in range(1, 11):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "KNeighbors(n_neighbors={0})".format(n_neighbors), metrics.f1_score(y_test, y_pred)
