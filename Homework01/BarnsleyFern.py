import numpy as np    # Handling numbers and arrays 
import matplotlib.pyplot as plt    # Plotting the fern 
import random    # Randomly selecting which transformation to apply 

num_points = 50000    # Number of iterations 

transforms = [    # Transformation matrices and probabilities

    {"a": 0, "b": 0, "c": 0, "d": 0.16, "e": 0, "f": 0, "p": 0.01},    # Stem
    {"a": 0.85, "b": 0.04, "c": -0.04, "d": 0.85, "e": 0, "f": 1.6, "p": 0.85},    # Main leaves
    {"a": 0.2, "b": -0.26, "c": 0.23, "d": 0.22, "e": 0, "f": 1.6, "p": 0.07},    # Left leaflet
    {"a": -0.15, "b": 0.28, "c": 0.26, "d": 0.24, "e": 0, "f": 0.44, "p": 0.07},    # Right leaflet
]

def apply_transform(x, y, transform):   # Defining the transformation function 
    new_x = transform["a"] * x + transform["b"] * y + transform["e"]
    new_y = transform["c"] * x + transform["d"] * y + transform["f"]
    return new_x, new_y

x, y = 0, 0    # Initial point
points_x = []   # Creating lists to store x points 
points_y = []   # Creating lists to store y points 

for _ in range(num_points):    # Each loop calculating a new point for the fern 
    r = random.random()    # Generating a random number between 0 and 1 
    cumulative_prob = 0    # Tracking the probability while selecting the transformation 
    for transform in transforms:    # Looping through the transformations 
        cumulative_prob += transform["p"]    # Adding the pobabilities 
        if r <= cumulative_prob:    # Applying the transformation if the random number r falls inside the transformation's range 
            x, y = apply_transform(x, y, transform)
            break
    points_x.append(x)   # Storing the new points 
    points_y.append(y)   # Storing the new points 

plt.figure(figsize=(6, 10))    # Creating the plot 
plt.scatter(points_x, points_y, s=0.1, color="black")    # Plotting all points as dots (size 0.1, black)
plt.axis("off")    # Removing the axis 
plt.title("Barnsley Fern")    # Adding a title 
plt.show()    # Displaying the final fern image 