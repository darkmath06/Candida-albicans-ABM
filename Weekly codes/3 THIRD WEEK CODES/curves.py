import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Original data
X = np.array([0, 4, 8, 16])

Y1 = np.array([0, 20, 60, 10])
Y2 = np.array([0, 20, 35, 90])

# Create smooth X values
X_smooth = np.linspace(X.min(), X.max(), 300)

# Cubic spline interpolation
cs1 = CubicSpline(X, Y1)
cs2 = CubicSpline(X, Y2)

Y1_smooth = cs1(X_smooth)
Y2_smooth = cs2(X_smooth)

# Plot
plt.plot(X_smooth, Y1_smooth, label="Curve 1")
plt.plot(X_smooth, Y2_smooth, label="Curve 2")

# Original points (optional, for reference)
plt.scatter(X, Y1)
plt.scatter(X, Y2)

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Smoothed Curves")
plt.legend()
plt.grid(True)

plt.show()