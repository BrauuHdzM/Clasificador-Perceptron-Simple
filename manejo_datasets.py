import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle

class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

class data_set:
	def __init__(self, validation_set, test_set):
		self.validation_set = validation_set
		self.test_set = test_set

def generate_train_test(file_name, pliegues):
	pd.options.display.max_colwidth = 200

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['target'],axis=1).values
	y = df['target'].values

	#Separa el corpus cargado en el DataFrame en el 50% para entrenamiento y el 50% para pruebas
	#~ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)


	#~ #Crea pliegues para la validación cruzada
	validation_sets = []
	kf = KFold(n_splits=pliegues)
	for train_index, test_index in kf.split(X_train):
	#~ #	print("TRAIN:", train_index, "\n",  "TEST:", test_index)
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))

	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)

	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set)

	return (my_data_set)

if __name__=='__main__':

	my_data_set_10 = generate_train_test('heart.csv', 3)

	i = 1
	for val_set in my_data_set_10.validation_set:
		np.savetxt("data_validation_train_<3>_<" + str(i) + ">.csv", val_set.X_train, delimiter=",", fmt="%s",
           header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments="")
		np.savetxt("data_test_<3>_<" + str(i) + ">.csv", val_set.X_test, delimiter=",", fmt="%s",
		   header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments="")
		np.savetxt("target_validation_train_<3>_<" + str(i) + ">.csv", val_set.y_train, delimiter=",", fmt="%s",
           header="target", comments="")
		np.savetxt("target_test_<3>_<" + str(i) + ">.csv", val_set.y_test, delimiter=",", fmt="%s",
           header="target", comments="")
		i = i + 1

