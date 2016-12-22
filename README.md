# Prediction-of-Class-of-Breast-Cancer
Implemented on Python using KNN classifier model.

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
df = pd.read_csv('/Users/Dinesh/Desktop/BDAP/Projects BDAP/Breast Cancer Data Set/breast-cancer-wisconsin.data.txt')


df.columns=('id','clump_thickness','uniform_cell_size',
'uniform_cell_shape','marginal_adhesion',
'single_epi_cell_size','bare_nuclei','bland_chromation',
'normal_nucleoli','mitoses','class')

df.replace('?',-99999, inplace=True)

df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([7,3,2,10,5,10,5,4,4])
prediction = clf.predict(example_measures)
print(prediction)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

