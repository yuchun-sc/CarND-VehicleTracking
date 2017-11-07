from sklearn import svm
import json

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

with open("./features_YCrCb_32.json") as f:
    raw_data = f.readlines()

X = []
y = []
for line in raw_data:
    line_string = json.loads(line)
    X.append(line_string["feature"])
    y.append(line_string["label"])
print len(X), len(y)

y = np.array(y)
X = np.array(X)

X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(y_train[y_train == 1].shape)
print(y_test[y_test == 1].shape)
#
# # clf = svm.SVC()
clf = svm.LinearSVC()
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#
with open('svm_model_linea_YCrCb_32.pkl', 'wb') as fid:
    pickle.dump(clf, fid)
