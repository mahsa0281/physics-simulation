import numpy as np
import random
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 5*x

def g(x):
    return 5*x - x**3
I_exact = -6.0
xs = np.linspace(0, 2, 5000)
g_values = g(xs)
g_max = np.max(g_values)

print(f"Estimated maximum of g(x) on [0,2]: {g_max:.3f}")

N_values = [50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
errors = []
estimates = []

for N in N_values:
    N_S = 0
    for _ in range(N):
        x_rand = random.uniform(0, 2)
        y_rand = random.uniform(0, g_max)
        if y_rand <= g(x_rand):
            N_S += 1
    
    I_g = 2.0 * g_max * (N_S / N)
    I_f = -I_g
    error = I_f - I_exact
    errors.append(error)
    estimates.append(I_f)
    
    print(f"N = {N:<6d}  I_est = {I_f:8.4f}  Error = {error:8.4f}")

plt.figure()
plt.plot(N_values, errors, marker='o')
plt.xlabel("Number of throws (N)")
plt.ylabel(r"$I_{\mathrm{est}} - I_{\mathrm{exact}}$")
plt.title("Hit-or-Miss (Monte Carlo) Error")
max_error = max(abs(e) for e in errors)
plt.ylim(-max_error, max_error)
plt.axhline(0, color='gray', linestyle='--')
plt.show()