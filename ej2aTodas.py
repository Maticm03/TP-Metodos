import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
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

# Calcular los coeficientes para los splines cúbicos
spline = CubicSpline(x_points, y_points)

# Crear puntos para graficar la función original y las interpolaciones
x_vals = np.linspace(-3, 3, 400)
y_original = f(x_vals)
y_linear = linear_interp(x_vals)
y_spline = spline(x_vals)

# Crear subplots para cada método de interpolación with more space
fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True, gridspec_kw={'hspace': 0.5})

# Subplot 1: Interpolación Lineal
axs[0].plot(x_vals, y_original, label='Función Original')
axs[0].plot(x_vals, y_linear, label='Interpolación Lineal', linestyle='dashed')
axs[0].scatter(x_points, y_points, color='red', label='Puntos de Colocación')
axs[0].set_title('Interpolación Lineal para f(x)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x)')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Polinomio de Lagrange
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

y_lagrange = np.array([lagrange_polynomial(x, x_points, y_points) for x in x_vals])
axs[1].plot(x_vals, y_original, label='Función Original')
axs[1].plot(x_vals, y_lagrange, label='Polinomio de Lagrange', linestyle='dashed')
axs[1].scatter(x_points, y_points, color='red', label='Puntos de Colocación')
axs[1].set_title('Interpolación de Lagrange para f(x)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('f(x)')
axs[1].legend()
axs[1].grid(True)

# Subplot 3: Splines Cúbicos
axs[2].plot(x_vals, y_original, label='Función Original')
axs[2].plot(x_vals, y_spline, label='Splines Cúbicos', linestyle='dashed')
axs[2].scatter(x_points, y_points, color='red', label='Puntos de Colocación')
axs[2].set_title('Interpolación con Splines Cúbicos para f(x)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('f(x)')
axs[2].legend()
axs[2].grid(True)

# Ajustar la disposición de los subplots y mostrar el gráfico completo
plt.tight_layout()
plt.show()
