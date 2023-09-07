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

# Interpolación de Lagrange
def lagrange_interpolation(points, target_x, target_y):
    n = int(np.sqrt(len(points)))
    result = 0.0

    for i in range(n):
        for j in range(n):
            term = points[i * n + j][2]
            for k in range(n):
                if k != i:
                    term *= (target_x - points[k * n + j][0]) / (points[i * n + j][0] - points[k * n + j][0])
            for k in range(n):
                if k != j:
                    term *= (target_y - points[i * n + k][1]) / (points[i * n + j][1] - points[i * n + k][1])
            result += term

    return result

lagrange_error = np.abs(f(target_X, target_Y) - lagrange_interpolation(points, target_X, target_Y))

# Interpolación de Splines
grid_x, grid_y = np.meshgrid(target_x, target_y)
grid_z_spline = griddata(np.array([(p[0], p[1]) for p in points]), np.array([p[2] for p in points]), (grid_x, grid_y), method='cubic')

spline_error = np.abs(f(target_X, target_Y) - grid_z_spline)

# Crear figuras para mostrar los errores absolutos
fig_lagrange_error = plt.figure()
ax_lagrange_error = fig_lagrange_error.add_subplot(111, projection='3d')

# Gráfico del error absoluto para Lagrange
ax_lagrange_error.plot_surface(target_X, target_Y, lagrange_error, cmap='viridis', alpha=0.5, label='Lagrange Error')
ax_lagrange_error.set_xlabel('X')
ax_lagrange_error.set_ylabel('Y')
ax_lagrange_error.set_zlabel('Absolute Error')
ax_lagrange_error.set_title('Lagrange Absolute Error')

fig_spline_error = plt.figure()
ax_spline_error = fig_spline_error.add_subplot(111, projection='3d')

# Gráfico del error absoluto para Splines
ax_spline_error.plot_surface(target_X, target_Y, spline_error, cmap='plasma', alpha=0.5, label='Spline Error')
ax_spline_error.set_xlabel('X')
ax_spline_error.set_ylabel('Y')
ax_spline_error.set_zlabel('Absolute Error')
ax_spline_error.set_title('Spline Absolute Error')

plt.show()
