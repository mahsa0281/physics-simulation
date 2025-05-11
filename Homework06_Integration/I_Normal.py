import numpy as np

N = 100000

x = np.random.uniform(0, 2, N)

height = np.exp(-x**2)

average_height = height.mean()

estimated_value = average_height * 2

print("Estimated Value: " , estimated_value)