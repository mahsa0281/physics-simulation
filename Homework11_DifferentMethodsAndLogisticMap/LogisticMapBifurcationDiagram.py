import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 100

def logistic_map(r, n_iterations, x0):
    x = x0
    last_values = np.zeros(100)

    for i in range(n_iterations):
        x = 4 * r * x * (1 - x)
        if i >= n_iterations - 100:
            last_values[i - (n_iterations - 100)] = x

    return last_values

r_values = np.arange(0.0, 1.0, 0.001)  
N = len(r_values)
x0 = 0.2  

bifurcation_data = np.zeros((N, 100))

for i in range(N):
    r = r_values[i]
    bifurcation_data[i] = logistic_map(r, 1000, x0)

plt.figure(figsize=(10, 6))
plt.plot(r_values, bifurcation_data, 'bo', markersize=0.3)
plt.title("Logistic Map Bifurcation Diagram")
plt.xlabel("r")
plt.ylabel("x* (long-term values)")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()