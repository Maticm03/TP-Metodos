import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from math import tanh, sin

# Define the original function
def f(x):
    return 0.05**abs(x) * np.sin(5*x) + np.tanh(2*x) + 2

# Define the number of data points
num_points = 15  # Change the number of data points to 15

# Define the points for interpolation
x_points = np.linspace(-3, 3, num_points)
y_points = f(x_points)

# Create a function for linear interpolation
linear_interp = interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')

# Create points for calculating errors
x_vals = np.linspace(-3, 3, 400)
y_original = f(x_vals)
y_linear = linear_interp(x_vals)

# Calculate errors for linear interpolation
error_linear = np.abs(y_original - y_linear)

# Create a function for Lagrange polynomial interpolation
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

# Calculate Lagrange interpolation
lagrange_interp = np.array([lagrange_polynomial(x, x_points, y_points) for x in x_vals])

# Calculate errors for Lagrange interpolation
error_lagrange = np.abs(y_original - lagrange_interp)

# Create a function for cubic spline interpolation
spline = CubicSpline(x_points, y_points)

# Calculate cubic spline interpolation
y_spline = spline(x_vals)

# Calculate errors for cubic spline interpolation
error_spline = np.abs(y_original - y_spline)

# Define the interval for computing the maximum error [-1, 1]
interval_start = -1
interval_end = 1

# Find the indices corresponding to the interval [-1, 1]
interval_indices = np.where((x_vals >= interval_start) & (x_vals <= interval_end))

# Calculate and print the maximum error for each interpolation method within the interval
max_error_linear = np.max(error_linear[interval_indices])
max_error_lagrange = np.max(error_lagrange[interval_indices])
max_error_spline = np.max(error_spline[interval_indices])

print(f"Maximum Error for Linear Interpolation within [{interval_start}, {interval_end}]: {max_error_linear:.4f}")
print(f"Maximum Error for Lagrange Interpolation within [{interval_start}, {interval_end}]: {max_error_lagrange:.4f}")
print(f"Maximum Error for Cubic Spline Interpolation within [{interval_start}, {interval_end}]: {max_error_spline:.4f}")

# Create separate plots for linear interpolation, Lagrange interpolation, and cubic spline interpolation
plt.figure(figsize=(10, 12))

plt.subplot(3, 1, 1)  # Create the first subplot for linear interpolation
plt.plot(x_vals, error_linear, label='Linear Interpolation Error')
plt.title('Error for Linear Interpolation')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.subplot(3, 1, 2)  # Create the second subplot for Lagrange interpolation
plt.plot(x_vals, error_lagrange, label='Lagrange Interpolation Error', color='green')
plt.title('Error for Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.subplot(3, 1, 3)  # Create the third subplot for cubic spline interpolation
plt.plot(x_vals, error_spline, label='Cubic Spline Interpolation Error', color='orange')
plt.title('Error for Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.tight_layout()
plt.show()
