import numpy as np    # NumPy library, used for efficient numerical operations
import matplotlib.pyplot as plt    # Matplotlib, used for visualisation
from scipy import stats     # stats module from SciPy, providing statistical functions
import random     # Python's random module, used for random choices

L = 200     # System size in 1D (200 columns where particles can be deposited)
num_layers = 3     # Number of deposition layers
steps_per_layer = 500 * L     # Ensuring that on average, each of the L sites receives 50 deposition events per layer
T = num_layers * steps_per_layer     # Total deposition steps across all layers
record_interval = 1000     # Frequency for recording data (e.g., every 1000 deposition steps)

heights = np.zeros(L, dtype=int)     # Creating an array of zeros of length L for initial heights (zero particles)
snapshots = [np.zeros(L, dtype=int)]     # Initialising a list to store snapshots of the height profile (starting with the initial state)

times = []     # Creating an list to store the deposition steps at which data is recorded
avg_heights_over_time = []    # Creating an empty list to store the average height at each recorded time
roughness_over_time = []     # Creating an empty list to store the roughness (standard deviation of heights) at each recording

t = 0    # Initialising a counter t to track the number of deposition steps

for layer in range(num_layers):     # Looping over each deposition layer (3 layers)
    for _ in range(steps_per_layer):     # Looping for the specified number of deposition steps per layer
        i = np.random.randint(0, L)    # Randomly choose a deposition site (index between 0 and L-1)
        
        left_index = (i - 1) % L    # Applying periodic boundary conditions: Left neighbour (wraps around: index -1 becomes L-1)
        right_index = (i + 1) % L    # Applying periodic boundary conditions: Right neighbour (wraps around: index L becomes 0)
        
        if heights[left_index] < heights[i] or heights[right_index] < heights[i]:     # Checking if either neighbour is lower than the chosen site
            left_height = heights[left_index]     # Getting the height of the left neighbour
            right_height = heights[right_index]     # Getting the height of the left neighbour
            
            if left_height < right_height:    # If the left neighbour is lower than the right neighbour,
                deposit_index = left_index     # Deposit particle at the left neighbour site
            elif right_height < left_height:    # If the right neighbour is lower than the left neighbour,
                deposit_index = right_index     # Deposit particle at the right neighbour site
            else:    # If both neighbours are equally lower than the chosen site,
                deposit_index = random.choice([left_index, right_index])     # Randomly choose one of them

            heights[deposit_index] += 1     # Deposit the particle at the chosen neighbour site
        else:     # If neither neighbour is lower,
            heights[i] += 1     # Deposit at the originally chosen site
        
        t += 1     # Increasing the deposition step counter
        
        if t % record_interval == 0:     # Checking if the current step t is a multiple of the record interval
            times.append(t)     # Appending the current time step t to the times list
            avg_heights_over_time.append(np.mean(heights))      # Recording the average height (mean of heights array)
            roughness_over_time.append(np.std(heights))      # Recording the roughness (standard deviation of heights)
    
    snapshots.append(heights.copy())     # Saving a snapshot of the current height profile after completing each layer

plt.figure(figsize=(8, 5))     # Creating a new figure for plotting
x = np.arange(L)     # Creating an array of positions from 0 to L-1 (substrate positions)
colors = ["blue", "grey", "red"]     # Defining a list of colours for visualising each layer

for i in range(num_layers, 0, -1):     # Loop from num_layers to 1 in steps of -1
    plt.fill_between(x, 0, snapshots[i], color=colors[num_layers - i], step='pre')     # Filling the area from y=0 up to the snapshot height profile, using the corresponding colour

plt.xlim(0, L)     # Setting the x-axis limits from 0 to L
plt.ylim(0, np.max(heights)*1.01)     # Setting the y-axis limits from 0 to slightly above the maximum height
plt.xlabel('Position along the substrate')     # Labeling for the x-axis
plt.ylabel('Height')     # Labeling for the y-axis
plt.title("1D Ballistic Deposition with Relaxation")     # Plotting the title
plt.show()     # Displaying the overlay plot

times_arr = np.array(times)     # Converting the times list to a NumPy array
avg_heights_arr = np.array(avg_heights_over_time)     # Converting the average heights list to a NumPy array
roughness_arr = np.array(roughness_over_time)     # Converting the roughness list to a NumPy array

reg_avg = stats.linregress(times_arr, avg_heights_arr)  # Performing linear regression on average height vs. time data

plt.figure(figsize=(8, 5))     # Creating a new figure for the average height plot
plt.plot(times_arr, avg_heights_arr, 'b.', label="Average Height Data")    # Plotting the average height data as blue dots
plt.plot(times_arr, reg_avg.slope * times_arr + reg_avg.intercept, 'r-', 
         label=f"Fit: slope={reg_avg.slope:.4f}, r={reg_avg.rvalue:.4f}")     # Plotting the the regression line in red
plt.xlabel("Time (deposition steps)")    # Labeling for the x-axis
plt.ylabel("Average Height")     # Labeling for the y-axis
plt.title("Average Height vs Time")     # Plotting the title
plt.legend()    # Adding a legend
plt.show()    # Displaying the plot

log_times = np.log10(times_arr)     # Transforming time data to logarithmic scale (base 10)
log_roughness = np.log10(roughness_arr)     # Transform roughness data to logarithmic scale (base 10)
reg_rough_log = stats.linregress(log_times, log_roughness)     # Performing linear regression on log-log transformed data

roughness_fit = 10 ** (reg_rough_log.slope * np.log10(times_arr) + reg_rough_log.intercept)     # Converting regression back to original scale

plt.figure(figsize=(8, 5))     # Creating a new figure for the roughness plot
plt.loglog(times_arr, roughness_arr, 'b.', label="Roughness Data")  # Plotting the roughness data on a log-log scale as blue dots
plt.loglog(times_arr, roughness_fit, 'r-', 
           label=f"Fit: slope={reg_rough_log.slope:.4f}, r={reg_rough_log.rvalue:.4f}")  # Plotting the regression line on log-log scale
plt.xlabel("Time (deposition steps)")     # Labeling for the x-axis
plt.ylabel("Roughness (Std. Deviation)")      # Labeling for the y-axis
plt.title("Roughness vs Time (Log-Log Scale)")      # Plotting the title
plt.legend()    # Adding a legend
plt.show()    # Displaying the plot