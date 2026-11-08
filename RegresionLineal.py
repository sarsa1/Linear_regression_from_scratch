import matplotlib 
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('dataset_alturas_pesos.csv')

X = df['Altura'].values
y = df['Peso'].values

X_ones = np.c_[np.ones(X.size), X] # Operacion necesaria para saber los parametros de la ecuacion Y = B0 + B1*X
#Lo que hace es transformar el vector alturas en una matriz alturas con un atributo extra que es 1
#theta = (X^T X)^-1 @ X^T @ y se trata de esa ecuacion normal
#Ahora nos falta la pendiente.

theta = np.linalg.inv(X_ones.T @ X_ones) @ X_ones.T @ y # El arroba s la multiplicacion de matrices

#theta = (X^T X)^{-1} \cdot X^T y$ se trata de esa ecuacion normal

print(theta) 
#[-101.12390488 (es el B0 es el BIAS el punto de partida)    1.00938673 (es el B1 la pendiente)] 

#Peso del usuario que no sabemos segun prediccion

weight = theta[0] + (theta[1] * 176) # 176 era la altura a predecir
print(weight) 

# Parte visual de matplot, generado por inteligencia artificial:
plt.scatter(X, y, color='blue', label='Datos reales')
weight_line = theta[0] + (theta[1] * X)
plt.plot(X, weight_line, color='red', label='Línea de regresión normal')
plt.scatter([176], [weight], color='green', marker='X', s=100, label='Predicción 176cm')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.grid(True)
plt.show()