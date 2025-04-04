import numpy as np
import matplotlib.pyplot as plt
import random

N_total = 100000
sequence = [random.randint(0, 9) for _ in range(N_total)]

extracted_digits = []
for i in range(len(sequence)):
    if sequence[i] == 4:
        extracted_digits.append(sequence[(i + 1) % len(sequence)])

frequency = [extracted_digits.count(i) for i in range(10)]
print("Frequency distribution of extracted digits:", frequency)

title_str = f'Frequency Distribution of Digits Following a 4 for N = {N_total}'

plt.figure()
plt.bar(range(10), frequency, tick_label=range(10))
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.title(title_str)
plt.show()

N_limit = 100000
T = np.array([])       
N_STD = np.array([])   

n = 50 

while n <= N_limit:
    numbers = np.array([random.randint(0, 9) for _ in range(n)])
    extracted = []
    for i in range(n):
        if numbers[i] == 4:
            extracted.append(numbers[(i + 1) % n])
    extracted = np.array(extracted)
    
    counts = np.zeros(10)
    for digit in extracted:
        counts[int(digit)] += 1
    
    std = np.sqrt(np.var(counts))
    
    if len(extracted) > 0:
        norm_std = std / len(extracted)
    else:
        norm_std = 0
    
    T = np.append(T, len(extracted))
    N_STD = np.append(N_STD, norm_std)
    
    n += n

print("Normalized STD values:", N_STD)
print("Number of extracted digits (T):", T)

log_T = np.log(T)
log_N_STD = np.log(N_STD)

plt.figure()
plt.plot(log_T, log_N_STD, 'bo', label='Data')

fit = np.polyfit(log_T, log_N_STD, 1)
m, b = fit[0], fit[1]
print("Slope for log(normalized STD) vs. log(T):", m, "// y-intercept:", b)

x_fit = [log_T[0], log_T[-1]]
y_fit = [b + m * log_T[0], b + m * log_T[-1]]
plt.plot(x_fit, y_fit, color='blue', label=r'Fitted line: $\frac{\sigma(N)}{N}$')

plt.grid(True)
plt.xlabel("log(Number of Extracted Values)")
plt.ylabel("log(Normalized STD)")
plt.title("Log–Log Plot: Normalized STD of Extracted Digits vs. Sample Size")
plt.legend()
plt.show()