import numpy as np   # NumPy library, used for efficient numerical operations
import matplotlib.pyplot as plt   # Matplotlib, used for visualisation
from scipy import stats   # stats module from the SciPy library, providing statistical functions

L = 200    # System size in 1D (200 columns where particles can be deposited)
N = 1000   # Total number of deposition events
angle = np.pi/3    # Deposition angle (60° from the vertical)
tan_angle = np.tan(angle)    # Calculating the tangent of the deposition angle

heights = np.zeros(L, dtype=int)    # Creating an array of zeros with length L to represent the initial height (zero particles) at each position along the substrate
base = np.arange(L)    # Array of positions along the substrate (used for plotting)

for j in range(N):   # Looping over each deposition event
    rnd = np.random.randint(0, L)    # Randomly selecting a column where the particle enters the system

    for i in range(rnd + 1):      # Simulating the particle's path by iterating from the random column towards the left (0-index)
        height_ball_i = heights[rnd] + ((rnd - i) * tan_angle)     # Calculating the effective height of the particle at column i along its trajectory

        if height_ball_i <= heights[i]:      # Checking if the particle collides with column i (including collisions at mid-column)
            heights[i] += 1    # Increasing the height at the collision column by one unit
            break    # Particle is deposited; exit the inner loop

plt.figure(figsize=(8, 5))    # Creating a new figure for plotting
for i in range(L):    # Looping through each column 
    plt.plot([base[i], base[i]], [0, heights[i]], 'b')    # Plotting each column as a vertical blue line

av_height = np.mean(heights)    # Computing the average height of the substrate
std_height = np.std(heights)    # Computing the roughness (standard deviation of the heights)

plt.xlabel('Position along the substrate')    # Setting the x-axis label
plt.ylabel('Height')    # Setting the y-axis label
plt.title("1D Angled Ballistic Deposition")    # Adding a title to the plot
plt.xlim(0, L)    # Setting the limits of the x-axis from 0 to L
plt.ylim(0, np.max(heights)*1.01)    # Setting the y-axis limits from 0 to slightly above the maximum height
plt.show()    # Displaying the plot