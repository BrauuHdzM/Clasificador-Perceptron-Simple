import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from  sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def activation_function (predicted_values):
	threshold_values = []
	
	for value in predicted_values:
		if value <0:
			threshold_values.append(0)
		else:
			threshold_values.append(1)
	
	return (threshold_values)
	
def weight_adjustment(y_predicted, y_train, weights, x_train):
	for i in range(len(y_train)):
		# ~ print ('y_train: {} - y_predicted: {}'.format(y_train[i], y_predicted[i]))
		error = y_train[i] - y_predicted[i]
		# ~ print ('error: ', error)
		if error != 0:
			weights += np.sum([weights,np.multiply(x_train[i], error)], axis=0)
			# ~ print ('weights: {}'.format(weights)) 
	return (weights)
	
	
#if __name__ == "__main__":
	# print("Number of epochs;Weights;Average Accuracy")
	
	# #para cada fold
	# for foldi in range(1,4,1):
		
	# 	fold=str(foldi)
	# 	# ~ print("El fold actual es: ",foldi)
	# 	nfile = 'data_validation_train__3___'+fold+'_'+'.csv'
	# 	nfiley = 'target_validation_train__3___'+fold+'_'+'.csv'
		
	# 	x_train = pd.read_csv(nfile, engine='python')
	# 	y = pd.read_csv(nfiley, engine='python')
	# 	y_train = y['target'].values
		
		
	# 	#weights = np.array([]).append(len())
		
	# 	weights=np.zeros(13)
	# 	# ~ print(weights)
	# 	x_standard_scaler = preprocessing.StandardScaler().fit_transform(x_train)
		
	# 	epochs = [2,5,8,10]
	# 	for epoch in epochs:
	# 		for i in range (epoch):
	# 			# ~ print ('----------------Iteración ', i, ' -------------------\n')
	# 			weight_sums = np.dot(x_standard_scaler,weights.T)# Hacemos la transpuesta, aunque al ser los pesos un vector de una dimensión python no requiere hacer la transpuesta para resolver la multiplicación porque lo hace mediante el producto punto u.v = u1.v1 + u2.v2 + ... + un.vn
	# 			# ~ print ('weight_sums:\n', weight_sums)
	# 			y_predicted = activation_function(weight_sums)
	# 			# ~ print ('y_predicted:', y_predicted)
	# 			# ~ print ('y_true: ', y_train)
				
	# 			# ~ print ('accuracy: ', accuracy_score(y_train, y_predicted))
				
	# 			weights = weight_adjustment(y_predicted, y_train, weights, x_standard_scaler)
		
	# 		# ~ print ('final weights :', weights)
	# 		# ~ print ('final accuracy: ', accuracy_score(y_train, y_predicted))
	# 		# ~ w=" ".join(map(str, weights))
	# 		# ~ print(w)
	# 		# ~ print(""+epochs+";"+w+";"+accuracy_score(y_train, y_predicted))
			
	# 		print(epoch, end='')
	# 		print(";", end='')
	# 		# ~ print(weights, end='')
	# 		for w in weights:
	# 			print(w, end='')
	# 			print(",", end='')
	# 		print(";", end='')
	# 		print(accuracy_score(y_train, y_predicted))
	
###########################################################SEGUNDA PARTE############################################################################################################
#utilizando el 70% de los datos para entrenar y el 30% para validar 
# Path: perceptrón_simple.py
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from  sklearn import preprocessing

def activation_function (predicted_values):
	threshold_values = []
	
	for value in predicted_values:
		if value <0:
			threshold_values.append(0)
		else:
			threshold_values.append(1)
	
	return (threshold_values)

def weight_adjustment(y_predicted, y_train, weights, x_train):
	for i in range(len(y_train)):
    		# ~ print ('y_train: {} - y_predicted: {}'.format(y_train[i], y_predicted[i]))
		error = y_train[i] - y_predicted[i]
		# ~ print ('error: ', error)
		if error != 0:
			weights += np.sum([weights,np.multiply(x_train[i], error)], axis=0)
			# ~ print ('weights: {}'.format(weights)) 
	return (weights)

if __name__ == "__main__":
	#utilizando el 70% de los datos para entrenar y el 30% para validar

	print("Epocas;Pesos;Accuracy")

	#cargo los datos de entrenamiento del dataset heart
	df = pd.read_csv("heart.csv", sep=',', engine='python')
	X = df.drop(['target'],axis=1).values
	y = df['target'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)

	#probamos con 4 epocas
	epochs = [2,5,8,10]
	for epoch in epochs:
		weights=np.zeros(13)
		# ~ print(weights)
		x_standard_scaler = preprocessing.StandardScaler().fit_transform(X_train)

		for i in range (epoch):
			# ~ print ('----------------Iteración ', i, ' -------------------\n')
			weight_sums = np.dot(x_standard_scaler,weights.T)
			# ~ print ('weight_sums:\n', weight_sums)
			y_predicted = activation_function(weight_sums)
			# ~ print ('y_predicted:', y_predicted)
			# ~ print ('y_true: ', y_train)

			# ~ print ('accuracy: ', accuracy_score(y_train, y_predicted))

			weights = weight_adjustment(y_predicted, y_train, weights, x_standard_scaler)

		# ~ print ('final weights :', weights)
		# ~ print ('final accuracy: ', accuracy_score(y_train, y_predicted))
		# ~ w=" ".join(map(str, weights))
		# ~ print(w)
		# ~ print(""+epochs+";"+w+";"+accuracy_score(y_train, y_predicted))

		print(epoch, end='')
		print(";", end='')
		# ~ print(weights, end='')
		for w in weights:
			print(w, end='')
			print(",", end='')
		print(";", end='')
		print(accuracy_score(y_train, y_predicted))

		#validamos con el 30% de los datos
		x_standard_scaler = preprocessing.StandardScaler().fit_transform(X_test)
		weight_sums = np.dot(x_standard_scaler,weights.T)
		y_predicted = activation_function(weight_sums)
		print("Accuracy validación: ", accuracy_score(y_test, y_predicted))

#usando matplot visualizaremos la matriz de confusión
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)
print(cm)

#visualizamos la matriz de confusión
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

#imprimir las metricas de precision, recall, f1-score y confusion matrix
print("Precision: ", precision_score(y_test, y_predicted))
print("Recall: ", recall_score(y_test, y_predicted))
print("F1-score: ", f1_score(y_test, y_predicted))
print("Confusion matrix:\n ", confusion_matrix(y_test, y_predicted))

