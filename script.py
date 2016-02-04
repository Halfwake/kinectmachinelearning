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
for datum in data:
	yi = datum['label']
	if yi == 2:
		continue
	y.append(yi)
	Xi = []
	features = datum['jointPositions']['jointPositionDict']
	collection = ['HipCenter', 'HipLeft', 'HipRight', 'KneeRight', 'KneeLeft', 'WristRight', 'WristLeft', 'HandRight', 'HandLeft', 'ElbowRight', 'ElbowLeft']
	for joint in collection:
		xj = features[joint].values()
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


	


