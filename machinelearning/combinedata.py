import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.externals import joblib
import pandas as pd
from tabulate import tabulate

class Combine_Data:
	def get_data():
		finaldat=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
		addr='C:/Users/G50/Desktop/emotiv/data';
		clas=['tc','b','m','n']
		lab=['1','2','3','0']
		clsr=[22,20,6,1]
		for i in range(0,4):
			newaddr=addr+"/"+str(clas[i])+"/"+str(clas[i])
			for j in range(1,clsr[i]+1):
				address=newaddr+str(j)+".csv"
				dat=pd.read_csv(address)
				dat=dat.values
				dat=np.array(dat[0:],dtype=np.float)
				b = np.zeros((100,15))
				b[:,:-1] = dat[0:100,:]
				b[0:50,14]=0
				b[51:100,14]=lab[i]
				dat2=b
				dat3=dat2[0:100,:]
				finaldat=np.concatenate((finaldat,dat3))
		return finaldat

	def write_data(finaldat):
		finaldat2=pd.DataFrame(data=finaldat)
		finaldat2.to_csv('C:/Users/G50/Desktop/emotiv/dataset.csv')

	data=get_data()
	write_data(data)
