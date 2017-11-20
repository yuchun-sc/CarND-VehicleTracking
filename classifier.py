from sklearn import svm
import json

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


class ClassifierTraining:
	def __init__(self, feature_file, model_file):
		self.feature_file = feature_file
		self.model_file = model_file

	def train(self):
		self._load_file_to_feature()

		print "training classifier"
		X, y = shuffle(self.X, self.y, random_state=0)

		X, X_scaler = self.normalize(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

		print X_train.shape, X_test.shape
		# # clf = svm.SVC()
		clf = svm.LinearSVC()
		clf.fit(X_train, y_train)

		print "train score:", clf.score(X_train, y_train)
		print "test  score:", clf.score(X_test, y_test)

		output_model = {
			"model": clf,
			"scaler": X_scaler
		}

		with open(self.model_file, 'wb') as fid:
			pickle.dump(output_model, fid)


	def normalize(self, X):
		X_scaler = StandardScaler().fit(X)
		result = X_scaler.transform(X)
		return result, X_scaler

	def _load_file_to_feature(self):
		print "loading feature from file"
		with open(self.feature_file) as f:
			raw_data = f.readlines()

		X = []
		y = []
		for line in raw_data:
			line_string = json.loads(line)
			X.append(line_string["feature"])
			y.append(line_string["label"])

		self.y = np.array(y)
		self.X = np.array(X)

