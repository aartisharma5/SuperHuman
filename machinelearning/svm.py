import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.externals import joblib
import pandas as pd
from tabulate import tabulate

class Classify_action:
	
	clf=SVC()
	def __init__(self,X,y):
		self.X=X
		self.y=y

	def get_data():
		data=pd.read_csv('C:/Users/G50/Desktop/emotiv/dataset.csv')
		return data
		
	def train_classifier(inp_data,inp_labels):
		clf = SVC()
		clf.fit(inp_data, inp_labels) 
		SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
		    max_iter=-1, probability=False, random_state=None, shrinking=True,
		    tol=0.001, verbose=False)
		joblib.dump(clf, 'filename.pkl') 
		print("done")


	def predict_class(inputsample):
		clf = joblib.load('filename.pkl') 
		return clf.predict(inputsample)

	#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	#y = np.array([1, 1, 3, 2])

	dat=get_data()
	dat=dat.values
	datarr=np.array(dat,dtype=np.float)
	datarr=np.asmatrix(datarr)
	datarr=dat[1:,1:]
	np.random.shuffle(datarr)
	print(datarr[1:5,:])
	inp_data=datarr[0:4000,0:14]
	inp_labels=datarr[0:4000:,14]
	test_data=datarr[4000:,0:14]
	test_labels=datarr[4000:,14]
	inp_labels=np.ravel(inp_labels,order='C')
	test_labels=np.ravel(test_labels,order='C')
	print(inp_labels)
	train_classifier(inp_data,inp_labels)
	test_out=predict_class(test_data)
	mae=sum(abs(test_out-test_labels))/np.size(test_labels)
	print(mae)
