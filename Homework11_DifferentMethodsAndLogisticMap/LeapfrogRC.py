import numpy as np
import matplotlib.pyplot as plt

R = 1
C = 1

def f(x):
    return -x / (R * C)

def numeric_step(x_0, h, T_f):
    time = np.arange(0, T_f, h)  
    x = x_0                      
    x_pp = x_0 - f(x)*h          
    x_p = x_0                    
    N = len(time)
    x_arr = np.zeros(N, dtype=np.float16)

    for n in range(N):
        x = x_pp + 2 * h * f(x_p)   
        x_pp = x_p                 
        x_p = x                  
        x_arr[n] = x                

    return x_arr, time

def analetic_solve(x_0, t):
    return x_0 * np.exp(-t / (R * C))

T_f = 7
x_0 = 1.
h = 0.01
x, t = numeric_step(x_0, h, T_f)

plt.plot(t, x, 'b', label='Numerical Calculation')
plt.plot(t, analetic_solve(x_0, t), 'r', label='Analytical Solution')
plt.title('Charge vs Time (Leapfrog Method)')
plt.xlabel('Time (t)')
plt.ylabel('Charge q(t)')
plt.legend()
plt.grid(True)
plt.show()