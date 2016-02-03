import json
import numpy as np
from pprint import pprint
from sklearn import svm
from sklearn import cross_validation

json_data = open('data/classifiedSitting.csv').read()
data = json.loads(json_data)
# pprint(data[0])
X = []
y = []
for i in range(len(data)):
	yi = data[i]['label']
	if yi == 2:
		continue
	y.append(yi)
	Xi = []
	features = data[i]['jointPositions']['jointPositionDict']
	collection = ['HipCenter', 'HipLeft', 'HipRight', 'KneeRight', 'KneeLeft', 'WristRight', 'WristLeft', 'HandRight', 'HandLeft', 'ElbowRight', 'ElbowLeft']
	for j in range(len(collection)):
		xj = features[collection[j]].values()
		Xi = Xi + xj
	X = X + [Xi]
# train with svm
# clf = svm.SVC()
# clf.fit(X, y)
# predict
# clf.predict()

#cross-validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.4, random_state = 0)
clf = svm.SVC().fit(X_train, y_train)
print clf.score(X_test, y_test)


	


