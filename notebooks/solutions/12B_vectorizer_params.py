# CountVectorizer
vectorizer = CountVectorizer(min_df=10, ngram_range=(1, 3))
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

visualize_coefficients(clf, vectorizer.get_feature_names())

# TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 3))
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

visualize_coefficients(clf, vectorizer.get_feature_names())
