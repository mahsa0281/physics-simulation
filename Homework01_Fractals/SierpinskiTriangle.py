import numpy as np  # NumPy library provides efficient numerical operations
import matplotlib.pyplot as plt  # Matplotlib’s pyplot module is used for creating plots and visualisations

def subdivide_triangle(triangle):

    scale = 0.5  # Shrinking factor
    T1 = lambda pts: pts * scale  #  T1: Stays at the bottom-left
    T2 = lambda pts: pts * scale + np.array([0.5, 0])  # T2: Moves right by half the original width
    T3 = lambda pts: pts * scale + np.array([0.25, np.sqrt(3)/4])  # T3: Moves upwards to maintain equilateral shape
    return [T1(triangle), T2(triangle), T3(triangle)]   # Applying the transformations to the input triangle


triangle0 = np.array([   # Initialising the Sierpinski triangle with an equilateral triangle defined by 3 vertices

    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3) / 2]
])

triangles = [triangle0]  # Starting with a single triangle (Iteration 0)

iterations = 10  # Adjusted as needed 

fig, ax = plt.subplots(figsize=(8, 8))  # Creating a new figure and a set of subplots
fig.suptitle("Sierpinski Triangle: Click on the plot for the next iteration")  # Title for interactivity

def plot_triangles(triangles, ax, iteration):

    ax.clear()  # Clearing any existing plots on the axes
    for tri in triangles:
        tri_closed = np.vstack([tri, tri[0]])  # Closing the triangle by repeating the first vertex
        ax.fill(tri_closed[:, 0], tri_closed[:, 1], color='black', edgecolor='black')  # Filling the triangle in black
    ax.set_aspect('equal')  # Ensuring equal scaling for x and y axes so the triangles aren’t distorted
    ax.set_xticks([])  # Removing tick marks on the x-axis
    ax.set_yticks([])  # Removing tick marks on the y-axis
    ax.set_title(f"Iteration {iteration}", fontsize=12)  # Setting a title to show the current iteration
    plt.draw()  # Redrawing the plot so the changes are visible
    plt.waitforbuttonpress()  # Pausing execution until the user clicks or presses a key

plot_triangles(triangles, ax, 0) # Display the initial triangle (Iteration 0)


for i in range(1, iterations + 1): # Iteratively subdividing the triangles
    new_triangles = []  # Creating an empty list to store the new, smaller triangles
    for tri in triangles:
        sub_triangles = subdivide_triangle(tri)  # Subdividing the current triangle into 3 smaller triangles

        new_triangles.extend(sub_triangles)  # Adding these new triangles to the list
    triangles = new_triangles  # Updating the list of triangles for the next iteration 

    plot_triangles(triangles, ax, i)  # Ploting the triangles for the current iteration

plt.show()  # After all iterations, displaying the final plot and keep the window open