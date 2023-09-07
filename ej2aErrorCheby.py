import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from math import tanh, sin

# Define the original function
def f(x):
    return 0.05**abs(x) * np.sin(5*x) + np.tanh(2*x) + 2

# Define the number of data points
num_points = 10

# Create points using Chebyshev nodes within the interval [-3, 3]
chebyshev_points = np.cos((2 * np.arange(1, num_points + 1) - 1) * np.pi / (2 * num_points)) * 3

# Sort the Chebyshev nodes in ascending order
chebyshev_points.sort()

# Calculate the corresponding function values
y_points = f(chebyshev_points)

# Create a function for linear interpolation
linear_interp = interp1d(chebyshev_points, y_points, kind='linear', fill_value='extrapolate')

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
lagrange_interp = np.array([lagrange_polynomial(x, chebyshev_points, y_points) for x in x_vals])

# Calculate errors for Lagrange interpolation
error_lagrange = np.abs(y_original - lagrange_interp)

# Create a function for cubic spline interpolation
spline = CubicSpline(chebyshev_points, y_points)

# Calculate cubic spline interpolation
y_spline = spline(x_vals)

# Calculate errors for cubic spline interpolation
error_spline = np.abs(y_original - y_spline)

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
