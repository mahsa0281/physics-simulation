import numpy as np   # Handling numerical arrays (our points)
import matplotlib.pyplot as plt   # Plotting the final triangle 
import random   # Picking random numbers 

vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])    # Defining the triangle's three fixed corners

point = np.array([random.uniform(0, 1), random.uniform(0, 1)])   # Choosing an initial random point (not necessarily inside the triangle)
# The initial point doesn’t matter because the process will "pull" the points into the triangle!

num_iterations = 10000    # Setting the number of iterations 

points = []    # Preparing a list for storing the points we generate 

for _ in range(num_iterations):    # Looping to generate enough points 

    chosen_vertex = random.choice(vertices)    # Picking one of the fixed triangle corners randomly 
    
    point = (point + chosen_vertex) / 2     # Moving our current point halway toward the chosen vertex
    
    points.append(point)   # Storing the new point

points = np.array(points)   # Converting the points to NumPy array for plotting

plt.figure(figsize=(6, 6))   # Setting the plot size
plt.scatter(points[:, 0], points[:, 1], s=0.1, color="black")   # Scatter plot all the points 
plt.axis("equal")   # Keeping the aspect ratio equal 
plt.axis("off")   # Removing unnecessary axis labels 
plt.title("Sierpinski Triangle (Random)")   # Adding a title 
plt.show()   # Displaying the final fractal 