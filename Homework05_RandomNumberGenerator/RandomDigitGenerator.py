import numpy as numpy 
import random
import matplotlib.pyplot as plt

N = 10000
results = [random.randint(0, 9) for _ in range(N)]

frequencies = [results.count(i) for i in range(10)]

title_str = f'Frequency of Each Digit (0-9) for N = {N}'
plt.bar(range(10), frequencies, tick_label=range(10))
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.title(title_str)
plt.show()