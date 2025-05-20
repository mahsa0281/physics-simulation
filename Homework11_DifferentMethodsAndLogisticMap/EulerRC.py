import numpy as np
import matplotlib.pyplot as plt

R = 1  
C = 1  

def f(x, t):
    return -x / (R * C)

def euler_step(x_0, h, T_f):
    time = np.arange(0, T_f, h, dtype=np.float16)  
    N = len(time) 
    x_arr = np.zeros(N, dtype=np.float16)  
    x = np.float16(x_0) 
    for n in range(N):
        x_arr[n] = x  
        x += np.float16(f(x, 1) * h) 
    return x_arr, time

def analytic_solve(x_0, t):
    return x_0 * np.exp(-t / (R * C))

T_f = 1  
x_0 = 1.  

h = np.arange(0.001, 0.1, 0.0001, dtype=np.float16)
error_arr = np.zeros(len(h), dtype=np.float16)

for i in range(len(h)):
    x, t = euler_step(x_0, h[i], T_f)  
    error_arr[i] = abs(x[-1] - analytic_solve(x_0, T_f))

optimal_h = h[np.argmin(error_arr)]
print(f"Optimal step size h for Euler method: {optimal_h}")

plt.plot(h, error_arr, markersize=1)
plt.title('Euler Method: Error vs Step Size')
plt.xlabel('Step size (h)')
plt.ylabel('Error at final time')
plt.grid(True)
plt.show()