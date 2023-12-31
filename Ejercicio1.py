#%% Lineal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos de mediciones y ground truth desde los archivos CSV
mediciones_data = pd.read_csv('mediciones.csv', header=None, delimiter=' ')
groundtruth_data = pd.read_csv('groundtruth.csv', header=None, delimiter=' ')

# Extracción de columnas de coordenadas
x1_med = mediciones_data[0].values
x2_med = mediciones_data[1].values
x1_gt = groundtruth_data[0].values
x2_gt = groundtruth_data[1].values

# Definir los límites de las zonas
def limite_vertical(x1):
    return [3.6] * len(x1)

def limite_recta(x1):
    return 3.6 - 0.35 * x1

# Encontrar las intersecciones con los límites
x2_limite_vertical = limite_vertical(x1_med)
x2_limite_recta = limite_recta(x1_med)

# Gráfico de la trayectoria y los límites
plt.figure(figsize=(10, 6))
plt.plot(x1_med, x2_med, label='Trayectoria Interpolada', color='blue', marker='o', linestyle='-', linewidth=2)
plt.plot(x1_gt, x2_gt, label='Ground Truth', color='green', marker='x', linestyle='-', linewidth=2)
plt.axvline(x=10, color='red', linestyle='--', label='Límite Vertical')
plt.plot(x1_med, x2_limite_recta, label='Límite Recta', linestyle='--', color='purple')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Trayectoria del Tractor y Límites del Campo')
plt.legend()
plt.grid()
plt.xlim([min(min(x1_med), min(x1_gt))-1, max(max(x1_med), max(x1_gt))+1])
plt.ylim([min(min(x2_med), min(x2_gt))-1, max(max(x2_med), max(x2_gt))+1])
plt.show()

#%% Cálculo de intersecciones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Cargar los datos de mediciones y ground truth desde los archivos CSV
mediciones_data = pd.read_csv('mediciones.csv', header=None, delimiter=' ')
groundtruth_data = pd.read_csv('groundtruth.csv', header=None, delimiter=' ')

# Extracción de columnas de coordenadas
x1_med = mediciones_data[0].values
x2_med = mediciones_data[1].values
x1_gt = groundtruth_data[0].values
x2_gt = groundtruth_data[1].values

# Ordenar los datos de mediciones en función de x1_med
sorted_indices = np.argsort(x1_med)
x1_med_sorted = x1_med[sorted_indices]
x2_med_sorted = x2_med[sorted_indices]

# Crear el spline cúbico con los datos ordenados
spline = CubicSpline(x1_med_sorted, x2_med_sorted)

# Definir los límites de las zonas
def limite_vertical(x1):
    return [3.6] * len(x1)

def limite_recta(x1):
    return 3.6 - 0.35 * x1

# Encontrar las intersecciones con los límites
x1_intersections_vertical = np.array([3.9, 5.0, 6.6, 10.0])
x2_intersections_vertical = spline(x1_intersections_vertical)

# Gráfico de la trayectoria y los límites
plt.figure(figsize=(10, 6))
plt.plot(x1_med, x2_med, label='Trayectoria Interpolada', color='blue', marker='o', linestyle='-', linewidth=2)
plt.plot(x1_gt, x2_gt, label='Ground Truth', color='green', marker='x', linestyle='-', linewidth=2)
plt.axvline(x=10, color='red', linestyle='--', label='Límite Vertical')
plt.plot(x1_med, x2_limite_recta, label='Límite Recta', linestyle='--', color='purple')
plt.scatter(x1_intersections_vertical, x2_intersections_vertical, color='red', label='Intersecciones')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Trayectoria del Tractor y Límites del Campo')
plt.legend()
plt.grid()
plt.xlim([min(min(x1_med), min(x1_gt))-1, max(max(x1_med), max(x1_gt))+1])
plt.ylim([min(min(x2_med), min(x2_gt))-1, max(max(x2_med), max(x2_gt))+1])
plt.show()
#%% Intersecciones 2
