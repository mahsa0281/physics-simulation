import numpy as np

N = 100000

u = np.random.random(N)

x = -np.log(1 - u * (1 - np.exp(-2)))

p_x = np.exp(-x) / (1 - np.exp(-2))

average_height = np.exp(-x**2) / p_x

estimated_value = np.mean(average_height) 

print("Estimated Value:", estimated_value)
