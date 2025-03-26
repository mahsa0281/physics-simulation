import numpy as np
import matplotlib.pyplot as plt

max_time = 1000           
time_step = 20            
num_walkers = 500        
directions = [(1,0), (-1,0), (0,1), (0,-1)]  
times = []
r_values = []  

for t in range(0, max_time + 1, time_step):
    if t == 0:
        times.append(t)
        r_values.append(0.0)
        continue
    
    sum_r2 = 0.0
    
    for _ in range(num_walkers):
        x, y = 0, 0
        for _ in range(t):
            dx, dy = directions[np.random.randint(4)]
            x += dx
            y += dy
        sum_r2 += (x**2 + y**2)
    
    mean_r = np.sqrt(sum_r2 / num_walkers)
    
    times.append(t)
    r_values.append(mean_r)

times_nonzero = np.array(times[1:], dtype=float)     
r_values_nonzero = np.array(r_values[1:], dtype=float)

log_t = np.log(times_nonzero)
log_r = np.log(r_values_nonzero)

plt.figure(figsize=(7,5))
plt.plot(log_t, log_r, 'bo', markersize=3, label="Data")

coeffs = np.polyfit(log_t, log_r, 1)
slope, intercept = coeffs
fit_line = slope*log_t + intercept

plt.plot(log_t, fit_line, 'r--', label=f"Fit: slope={slope:.3f}")
plt.xlabel(r'$\text{Time } (t)$')         
plt.ylabel(r'$\sqrt{\langle r^2 \rangle}$')  
plt.title("2D Random Walk (Monte Carlo Method)")
plt.legend()
plt.grid(True)
plt.show()