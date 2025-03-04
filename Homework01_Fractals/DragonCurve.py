import numpy as np  # NumPy library provides efficient numerical operations
import matplotlib.pyplot as plt  # Matplotlib’s pyplot module is used for creating plots and visualizations

def iterate_dragon(turns):  # Function to generate the next sequence of turns for the Dragon Curve
    return turns + [+90] + [-t for t in reversed(turns)]  # New turns follow the rule: Reverse & flip signs

def compute_dragon_points(turns, step=1.0):  # Function to compute the Dragon Curve points based on turn sequence
    points = [np.array([0, 0]), np.array([1, 0])]  # Start with the first segment from (0,0) to (1,0)
    angle = 0  # Initial direction is along the positive x-axis
    current_point = np.array([1, 0])  # The last known point of the segment

    for turn in turns:  # Starting a loop to process each turning angle in the list turns
        angle -= turn  # Updating the direction (Negative means clockwise rotation)

        displacement = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * step # Computing a new displacement vector for the updated direction
        current_point = current_point + displacement  # Moving to the next point
        points.append(current_point)  # Appending the new point to the list

    return np.array(points)  # Converting the list of points to a NumPy array for efficient computation

iterations = 20  # Number of iterations to generate the Dragon Curve

turns = []  # Starting with no turns (just a single segment)

fig, ax = plt.subplots(figsize=(8, 8))  # Creating a figure and subplot for visualisation
fig.suptitle("Dragon Curve: Click to see next iteration")  # Setting a title for interactivity

points = compute_dragon_points(turns)  # Computing the points for the first iteration
ax.plot(points[:, 0], points[:, 1], color='blue')  # Ploting the initial line
ax.set_aspect('equal')  # Ensuring equal scaling of x and y axes
ax.set_xticks([])  # Removing x-axis ticks for a cleaner visualization
ax.set_yticks([])  # Removing y-axis ticks for a cleaner visualization
ax.set_title("Iteration 0")  # Setting the title for the first iteration

plt.draw()  # Rendering the initial plot
plt.waitforbuttonpress()  # Waiting for user input to proceed to the next iteration

for i in range(1, iterations + 1):  # Looping through the number of iterations
    turns = iterate_dragon(turns)  # Generating the new sequence of turns
    points = compute_dragon_points(turns)  # Computing the new points

    ax.clear()  # Clear the current plot before drawing the updated curve
    ax.plot(points[:, 0], points[:, 1], color='blue')  # Plot the updated Dragon Curve
    ax.set_aspect('equal')  # Maintain equal scaling of x and y axes
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_title(f"Iteration {i}")  # Updating the title with the current iteration

    plt.draw()  # Updating the plot with the new curve
    plt.waitforbuttonpress()  # Pausing execution until the user clicks to continue

plt.show()  # Displaying the final iteration and keep the window open