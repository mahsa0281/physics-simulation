import numpy as np

N = 1000000

x = np.random.uniform(0, 2, N)

height = x**3 - 5 * x 

average_height = height.mean() 

estimated_value = average_height * 2

print("Estimated Integral Value: " , estimated_value)