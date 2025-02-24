import numpy as np  # NumPy library provides efficient numerical operations
import matplotlib.pyplot as plt  # Matplotlib’s pyplot module is used for creating plots and visualisations

def iterate_koch(points):  # Defining a function named iterate_koch that takes the current curve as input
    new_points = []  # Creating an empty list to store the points of the new curve
    for i in range(len(points) - 1):  # Looping over each segment of the current curve. The curve is defined by a sequence of points, and each pair of consecutive points forms a segment.
        
        p1 = points[i]  # p1 is the starting point of the current segment
        p2 = points[i + 1]  # p2 is the ending point of the current segment
        
        pA = p1 + (p2 - p1) / 3.0  # pA is the point one-third along the segment
        pB = p1 + 2 * (p2 - p1) / 3.0  # pB is the point two-thirds along the segment
        
        angle = np.radians(-60)    # Converting -60 degrees to radians (using -60 to ensure the bump is constructed outward for the snowflake)
        rotation_matrix = np.array([    # Creating the rotation matrix for a -60-degree rotation
            [np.cos(angle), -np.sin(angle)],  # First row: transforming the x-component
            [np.sin(angle),  np.cos(angle)]   # Second row: transforming the y-component
        ])
        pC = pA + np.dot(rotation_matrix, (pB - pA))    # Applying the rotation matrix to the vector from pA to pB
        
        new_points.extend([p1, pA, pC, pB])    # Appending the computed points in the correct order

    new_points.append(points[-1])    # After processing all segments, appending the very last point of the original curve

    return np.array(new_points)     # Converting the list of new points into a NumPy array for easier mathematical manipulation in future iterations

# Initialising the snowflake with an equilateral triangle defined by three points (closed loop)
points = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3)/2],
    [0, 0]  # Closing the triangle by repeating the first point
])

iterations = 10  # Adjusted as needed

fig, ax = plt.subplots(figsize=(8, 8))  # Creating a new figure and a set of subplots
fig.suptitle("Koch Snowflake: Click on the plot for the next iteration")  # Settting a title for the plot that instructs the user about interactivity

ax.plot(points[:, 0], points[:, 1], color='blue')  # Ploting the initial triangle using the x and y coordinates from the points array
ax.set_aspect('equal')  # Ensuring that the x and y scales are equal, so the triangle is not distorted
ax.set_xticks([])  # Removing any tick marks on the x-axis
ax.set_yticks([])  # Removing any tick marks on the y-axis
ax.set_title("Iteration 0", fontsize=12)  # Setting a title for the initial triangle displayed on the plot
plt.draw()  # Redrawing the plot so that the initial triangle and title are updated on the screen
plt.waitforbuttonpress()  # Pausing the execution until the user clicks on the plot or presses a key

for i in range(1, iterations + 1):  # Starting an iterative process to build the Koch snowflake step by step from the initial triangle

    points = iterate_koch(points)  # Generating the next iteration of the Koch snowflake by processing the current set of points

    ax.clear()  # Clearing the current axes to prepare for drawing the updated curve

    ax.plot(points[:, 0], points[:, 1], color='blue')  # Ploting the updated Koch snowflake using the x and y coordinates from the points array

    ax.set_aspect('equal')  # Ensuring that the x and y scales are equal, so the snowflake is not distorted

    ax.set_xticks([])  # Removing any tick marks on the x-axis
    ax.set_yticks([])  # Removing any tick marks on the y-axis

    ax.set_title(f"Iteration {i}", fontsize=12)  # Setting a title for the current iteration displayed on the plot

    plt.draw()  # Redrawing the plot so that the new snowflake and title are updated on the screen.
    plt.waitforbuttonpress()  # Pausing the execution until the user clicks on the plot or presses a key.

plt.show()  # After completing all iterations, displaying the final plot and keeping the window open