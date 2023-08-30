import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definición de la función original
def f(x, y):
    return (
        0.7 * np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4) +
        0.45 * np.exp(-((9*x+1)**2)/9 - ((9*y+1)**2)/5) +
        0.55 * np.exp(-((9*x-6)**2)/4 - ((9*y-3)**2)/4) -
        0.01 * np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    )

# Función de interpolación de Lagrange en dos dimensiones
def lagrange_interpolation(points, target_x, target_y):
    n = int(np.sqrt(len(points)))
    result = 0.0

    for i in range(n):
        for j in range(n):
            term = points[i*n + j][2]
            for k in range(n):
                if k != i:
                    term *= (target_x - points[k*n + j][0]) / (points[i*n + j][0] - points[k*n + j][0])
            for k in range(n):
                if k != j:
                    term *= (target_y - points[i*n + k][1]) / (points[i*n + j][1] - points[i*n + k][1])
            result += term

    return result

# Generar puntos equiespaciados en el intervalo [-1, 1]
n = 10  # Número de puntos en cada dimensión
x_points = np.linspace(-1, 1, n)
y_points = np.linspace(-1, 1, n)
points = [(x, y, f(x, y)) for x in x_points for y in y_points]

# Puntos objetivo para la interpolación
target_x = np.linspace(-1, 1, 50)
target_y = np.linspace(-1, 1, 50)
target_X, target_Y = np.meshgrid(target_x, target_y)
target_Z = np.zeros_like(target_X)

# Interpolación de Lagrange
for i in range(len(target_x)):
    for j in range(len(target_y)):
        target_Z[i, j] = lagrange_interpolation(points, target_x[i], target_y[j])

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Gráfico de la función original
ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points], color='blue', label='Original Function')

# Gráfico del polinomio interpolante
ax.plot_surface(target_X, target_Y, target_Z, cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Interpolation with Lagrange')

# Agregar texto explicativo
ax.text2D(0.05, 0.95, "Original Function", transform=ax.transAxes, color='blue')
ax.text2D(0.05, 0.90, "Interpolated Polynomial", transform=ax.transAxes, color='orange')

plt.show()
