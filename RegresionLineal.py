import matplotlib 
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('dataset_alturas_pesos.csv')

X = df['Altura'].values
y = df['Peso'].values


#Importamos la librería para hacer el train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_ones = np.c_[np.ones(X_train.size), X_train] # Operacion necesaria para saber los parametros de la ecuacion Y = B0 + B1*X
#Lo que hace es transformar el vector alturas en una matriz alturas con un atributo extra que es 1
#theta = (X^T X)^-1 @ X^T @ y se trata de esa ecuacion normal
#Ahora nos falta la pendiente.

theta = np.linalg.inv(X_ones.T @ X_ones) @ X_ones.T @ y_train # El arroba s la multiplicacion de matrices

#theta = (X^T X)^{-1} \cdot X^T y$ se trata de esa ecuacion normal

print(theta) 
#[-101.12390488 (es el B0 es el BIAS el punto de partida)    1.00938673 (es el B1 la pendiente)] 

#Peso del usuario que no sabemos segun prediccion

weight = theta[0] + (theta[1] * X_test) # 176 era la altura a predecir
print(weight) 

mse = np.mean((y_test - weight)**2)

print(f'El MSE: {mse}')
# Parte visual de matplot, generado por inteligencia artificial:
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='orange', label='Test (real)')
plt.scatter(X_test, weight, color='green', marker='x', label='Test (predicción)')
plt.plot(X, theta[0] + theta[1] * X, color='red', label='Recta de regresión')

plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.grid(True)
plt.show()