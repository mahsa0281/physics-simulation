import numpy as np
import random
import matplotlib.pyplot as plt

N = 100000
results = [random.randint(0, 9) for _ in range(N)]
frequencies = [results.count(i) for i in range(10)]

title_str = f'Frequency of Each Digit (0-9) for N = {N}'
plt.figure()
plt.bar(range(10), frequencies, tick_label=range(10))
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.title(title_str)
plt.show()

N = 100000
T = np.array([])
normalized_STD = np.array([])
n = 10

while n <= N:
    numbers = np.array([random.randint(0, 9) for _ in range(n)])
    
    counts = np.zeros(10)
    for number in numbers:
        counts[int(number)] += 1
        
    std = np.sqrt(np.var(counts))
    n_std = std / n
    
    T = np.append(T, n)
    normalized_STD = np.append(normalized_STD, n_std)
    
    n += n  

log_T = np.log(T)
log_STD = np.log(normalized_STD)

plt.figure()
plt.plot(log_T, log_STD, 'bo', label='Data')

fit = np.polyfit(log_T, log_STD, 1)
m, b = fit[0], fit[1]
print("Slope for log(normalized STD) vs log(T):", m, "// y-intercept:", b)

plt.plot([log_T[0], log_T[-1]], [b + m*log_T[0], b + m*log_T[-1]], 
         color='blue', label=r'Fitted line: $\frac{\sigma (N)}{N}$')
plt.grid(True)
plt.xlabel("log(T)")
plt.ylabel("log(Normalized STD)")
plt.title("Log of Normalized STD with Respect to T (number)")
plt.legend()
plt.show()