#%% Lagrange 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import tanh, sin

# Definir la función original
def f(x):
    return 0.05**abs(x) * np.sin(5*x) + np.tanh(2*x) + 2

# Función para el polinomio de Lagrange
def lagrange_polynomial(x, x_points, y_points):
    n = len(x_points)
    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Definir los puntos equiespaciados de colocación
num_points = 7
x_points = np.linspace(-3, 3, num_points)
y_points = f(x_points)

# Crear puntos para graficar la función original y el polinomio de Lagrange
x_vals = np.linspace(-3, 3, 400)
y_original = f(x_vals)
y_lagrange = np.array([lagrange_polynomial(x, x_points, y_points) for x in x_vals])

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_original, label='Función Original')
plt.plot(x_vals, y_lagrange, label='Polinomio de Lagrange', linestyle='dashed')
plt.scatter(x_points, y_points, color='red', label='Puntos de Colocación')
plt.title('Interpolación de Lagrange para f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

#%% Splines
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import tanh, sin

# Definir la función original
def f(x):
    return 0.05**abs(x) * np.sin(5*x) + np.tanh(2*x) + 2

# Definir los puntos equiespaciados de colocación
num_points = 7
x_points = np.linspace(-3, 3, num_points)
y_points = f(x_points)

# Calcular los coeficientes para los splines cúbicos
spline = CubicSpline(x_points, y_points)

# Crear puntos para graficar la función original y los splines cúbicos
x_vals = np.linspace(-3, 3, 400)
y_original = f(x_vals)
y_spline = spline(x_vals)

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_original, label='Función Original')
plt.plot(x_vals, y_spline, label='Splines Cúbicos', linestyle='dashed')
plt.scatter(x_points, y_points, color='red', label='Puntos de Colocación')
plt.title('Interpolación con Splines Cúbicos para f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# %% Lineal 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import tanh, sin

# Definir la función original
def f(x):
    return 0.05**abs(x) * np.sin(5*x) + np.tanh(2*x) + 2

# Definir los puntos equiespaciados de colocación
num_points = 7
x_points = np.linspace(-3, 3, num_points)
y_points = f(x_points)

# Crear una función de interpolación lineal
linear_interp = interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')

# Crear puntos para graficar la función original y la interpolación lineal
x_vals = np.linspace(-3, 3, 400)
y_original = f(x_vals)
y_linear = linear_interp(x_vals)

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_original, label='Función Original')
plt.plot(x_vals, y_linear, label='Interpolación Lineal', linestyle='dashed')
plt.scatter(x_points, y_points, color='red', label='Puntos de Colocación')
plt.title('Interpolación Lineal para f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


# %%
