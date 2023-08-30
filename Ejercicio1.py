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
