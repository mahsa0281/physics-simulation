import numpy as np
import matplotlib.pyplot as plt
import random

num_samples = 10000
std_dev = 1

x_vals = np.array([])
y_vals = np.array([])

for _ in range(num_samples):
    u = random.random()  
    theta = random.uniform(0, 2 * np.pi)  
    x = np.sqrt(-2 * std_dev**2 * np.log(1 - u)) * np.cos(theta)
    y = np.sqrt(-2 * std_dev**2 * np.log(1 - u)) * np.sin(theta)
    x_vals = np.append(x_vals, x)
    y_vals = np.append(y_vals, y)

hist_counts, bin_edges, _ = plt.hist(x_vals, bins=30)
calc_std = np.std(x_vals)
calc_mean = np.average(x_vals)
x_range = np.linspace(min(x_vals), max(x_vals), 100)
gaussian_fit = np.exp(-0.5 * ((x_range - calc_mean) / calc_std)**2) * max(hist_counts)
plt.plot(x_range, gaussian_fit, 'r')
plt.xlabel("Gaussian Samples (X-axis)")
plt.ylabel("Frequency")
plt.title(f"Histogram of {num_samples} Gaussian Samples")
plt.show()

plt.scatter(x_vals, y_vals, s=3, marker='o')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title(f"Scatter Plot of {num_samples} Gaussian Pairs (X, Y)")
plt.show()