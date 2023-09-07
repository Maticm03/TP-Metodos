import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline

# Definición de la función original
def f(x, y):
    return (
        0.7 * np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4) +
        0.45 * np.exp(-((9*x+1)**2)/9 - ((9*y+1)**2)/5) +
        0.55 * np.exp(-((9*x-6)**2)/4 - ((9*y-3)**2)/4) -
        0.01 * np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    )

# Generate equiespaciados puntos en el intervalo [-1, 1]
n = 10  # Número de puntos en cada dimensión
x_points = np.linspace(-1, 1, n)
y_points = np.linspace(-1, 1, n)
points = [(x, y, f(x, y)) for x in x_points for y in y_points]

# Puntos objetivo para la interpolación
target_x = np.linspace(-1, 1, 50)
target_y = np.linspace(-1, 1, 50)
target_X, target_Y = np.meshgrid(target_x, target_y)

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Gráfico de la función original
ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points], color='blue', label='Original Function')

# Interpolación de Splines
grid_x, grid_y = np.meshgrid(target_x, target_y)
grid_z = griddata(np.array([(p[0], p[1]) for p in points]), np.array([p[2] for p in points]), (grid_x, grid_y), method='cubic')

# Gráfico del polinomio interpolante
ax.plot_surface(target_X, target_Y, grid_z, cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Interpolation with Spline')

# Agregar texto explicativo
ax.text2D(0.05, 0.95, "Original Function", transform=ax.transAxes, color='blue')
ax.text2D(0.05, 0.90, "Interpolated Spline", transform=ax.transAxes, color='orange')

plt.show()
