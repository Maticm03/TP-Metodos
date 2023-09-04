import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def find_intersection_points(x1_interp_med, x2_interp_med):
    """
    Find all intersection points between the measured curve and the analytical lines.

    Args:
        x1_interp_med (numpy.ndarray): Interpolated x1 coordinates for measurement data.
        x2_interp_med (numpy.ndarray): Interpolated x2 coordinates for measurement data.

    Returns:
        numpy.ndarray: Array of intersection points as (x1, x2) tuples.
    """
    x2_mag = -0.35 * x1_interp_med + 3.6

    # Find the indices where the x2_interp_med crosses the analytical lines
    crosses_magenta_line = np.where(np.diff(np.sign(x2_interp_med - x2_mag)))[0]

    intersection_points = []

    # Find intersection points with the magenta line
    for index in crosses_magenta_line:
        x1_point1, x2_point1 = x1_interp_med[index], x2_interp_med[index]
        x1_point2, x2_point2 = x1_interp_med[index + 1], x2_interp_med[index + 1]

        # Interpolate to find the exact intersection point
        t = (x2_mag[index] - x2_point1) / (x2_point2 - x2_point1)
        x1_intersection = (1 - t) * x1_point1 + t * x1_point2
        x2_intersection = (1 - t) * x2_point1 + t * x2_point2

        if 0 <= x1_intersection <= 10:
            intersection_points.append((x1_intersection, x2_intersection))

    # Find intersection points with the green line (x1=10)
    for i in range(len(x1_interp_med) - 1):
        if (x1_interp_med[i] <= 10 <= x1_interp_med[i + 1]) or (x1_interp_med[i + 1] <= 10 <= x1_interp_med[i]):
            t = (10 - x1_interp_med[i]) / (x1_interp_med[i + 1] - x1_interp_med[i])
            x2_intersection = (1 - t) * x2_interp_med[i] + t * x2_interp_med[i + 1]
            intersection_points.append((10, x2_intersection))

    return np.array(intersection_points)

def create_continuous_curve_from_csv(csv_groundtruth, csv_mediciones):
    try:
        groundtruth_data = np.loadtxt(csv_groundtruth, delimiter=' ')

        # Extract columns of coordinates for ground truth
        x1_gt = groundtruth_data[:, 0]
        x2_gt = groundtruth_data[:, 1]

        # Calculate chord length as the parameterization for ground truth
        chord_length_gt = np.sqrt(np.diff(x1_gt)**2 + np.diff(x2_gt)**2)
        chord_length_gt = np.insert(chord_length_gt, 0, 0)
        cumulative_length_gt = np.cumsum(chord_length_gt)

        # Create parametric interpolation functions for x and y for ground truth
        t_gt = np.linspace(0, 1, len(x1_gt))
        t_interp_gt = np.linspace(0, 1, 1000)  # Adjust the number of points as needed

        x1_interp_gt = interp1d(t_gt, x1_gt, kind='cubic')(t_interp_gt)
        x2_interp_gt = interp1d(t_gt, x2_gt, kind='cubic')(t_interp_gt)

        # Load the measurements data from the CSV file
        mediciones_data = np.loadtxt(csv_mediciones, delimiter=' ')

        # Extract columns of coordinates for measurements
        x1_med = mediciones_data[:, 0]
        x2_med = mediciones_data[:, 1]

        # Create parametric interpolation functions for x and y for measurements
        t_med = np.linspace(0, 1, len(x1_med))
        t_interp_med = np.linspace(0, 1, 1000)  # Adjust the number of points as needed

        x1_interp_med = interp1d(t_med, x1_med, kind='cubic')(t_interp_med)
        x2_interp_med = interp1d(t_med, x2_med, kind='cubic')(t_interp_med)

        # Create a continuous curve plot for ground truth with filled dots for data points
        plt.plot(x1_interp_gt, x2_interp_gt, 'b-', label='Ground Truth (Cubic Spline)', linewidth=2)
        plt.scatter(x1_gt, x2_gt, c='b', marker='o', s=20, label='Ground Truth Data')

        # Create a continuous curve plot for measurements with filled dots for data points
        plt.plot(x1_interp_med, x2_interp_med, 'r--', label='Measurements (Cubic Spline)', linewidth=2)
        plt.scatter(x1_med, x2_med, c='r', marker='o', s=50, label='Measurement Data')

        # Add vertical line at x1 = 10
        plt.axvline(x=10, color='g', linestyle='--', label='Vertical Line at x1=10')

        # Create x2 = -0.35*x1 + 3.6 line
        x1_line = np.linspace(x1_gt.min(), x1_gt.max(), 100)
        x2_line = -0.35 * x1_line + 3.6
        plt.plot(x1_line, x2_line, 'm-', label='x2 = -0.35*x1 + 3.6 Line')

        # Set y-axis limit to not show anything below 0 in the x2 axis
        plt.ylim(0, None)

        # Calculate and plot all intersection points between the measured line and the magenta and green lines
        intersection_points = find_intersection_points(x1_interp_med, x2_interp_med)
        intersection_points = np.array(intersection_points)
        plt.scatter(intersection_points[:, 0], intersection_points[:, 1], c='black', marker='o', s=50, label='Intersection Points')

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Continuous Curve Comparison with Lines and Intersection Points')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
csv_groundtruth = 'mnyo_ground_truth.csv'
csv_mediciones = 'mnyo_mediciones.csv'
create_continuous_curve_from_csv(csv_groundtruth, csv_mediciones)
