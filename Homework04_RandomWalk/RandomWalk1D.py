import numpy as np
import matplotlib.pyplot as plt

p = 0.5
q = 1 - p
max_steps = 500

def step_function(prob):
    rnd = np.random.rand()
    return 1 if rnd <= prob else -1

mean_positions = [] 
variances = []       
std_devs = []        
time_points = []     
step_counts = []    

t = 0
n = 0
increment = 10       
num_walkers = 1000    

while n <= max_steps:
    final_positions = []
    
    for _ in range(num_walkers):
        x = 0
        for _ in range(n):
            x += step_function(p)
        final_positions.append(x)
    
    emp_mean = np.mean(final_positions)
    emp_var = np.var(final_positions)  
    emp_std = np.sqrt(emp_var)
    mean_positions.append(emp_mean)
    variances.append(emp_var)
    std_devs.append(emp_std)
    time_points.append(t)
    step_counts.append(n)
    t += 1
    n += increment

plt.figure(figsize=(8, 5))
plt.plot(step_counts, mean_positions, 'bo', label='Empirical ⟨x(t)⟩')
plt.plot(step_counts, variances, 'ro', label='Empirical σ²')

theoretical_means = []
theoretical_vars = []

for n_i in step_counts:
    theory_mean = (p - q) * n_i
    theory_var = 4 * p * q * n_i
    theoretical_means.append(theory_mean)
    theoretical_vars.append(theory_var)

plt.plot(step_counts, theoretical_means, 'b--', label='Theoretical ⟨x(t)⟩')
plt.plot(step_counts, theoretical_vars, 'r--', label='Theoretical σ²')
plt.grid(True)
plt.xlabel('Number of Steps (n)')
plt.ylabel(r'$\langle x(t)\rangle \text{ and } \sigma^2$')
plt.legend()
plt.title("1D Random Walk: Empirical vs. Theoretical Results")
plt.show()