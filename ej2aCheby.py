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

# Create points for plotting the interpolated functions
x_vals = np.linspace(-3, 3, 400)
y_original = f(x_vals)
y_linear = linear_interp(x_vals)

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

# Create a function for cubic spline interpolation
spline = CubicSpline(chebyshev_points, y_points)

# Calculate cubic spline interpolation
y_spline = spline(x_vals)

# Create a single plot showing all three interpolated functions
plt.figure(figsize=(10, 6))

# Linear Interpolation
plt.subplot(3, 1, 1)
plt.plot(x_vals, y_original, label='Original Function', color='blue')
plt.plot(x_vals, y_linear, label='Linear Interpolation', linestyle='dashed', color='red')
plt.scatter(chebyshev_points, y_points, color='black', label='Chebyshev Points')
plt.title('Linear Interpolation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Lagrange Interpolation
plt.subplot(3, 1, 2)
plt.plot(x_vals, y_original, label='Original Function', color='blue')
plt.plot(x_vals, lagrange_interp, label='Lagrange Interpolation', linestyle='dashed', color='green')
plt.scatter(chebyshev_points, y_points, color='black', label='Chebyshev Points')
plt.title('Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Cubic Spline Interpolation
plt.subplot(3, 1, 3)
plt.plot(x_vals, y_original, label='Original Function', color='blue')
plt.plot(x_vals, y_spline, label='Cubic Spline Interpolation', linestyle='dashed', color='orange')
plt.scatter(chebyshev_points, y_points, color='black', label='Chebyshev Points')
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
