import numpy as np   # NumPy libarary, used for efficient numerical operations
import matplotlib.pyplot as plt   # Matplotlib, used for visualisation
from scipy import stats   # stats module from the SciPy library, providing statistical functions 

L = 200    # System size in 1D (200 columns where particles can be deposited)
num_layers = 3    # Number of deposition layers
steps_per_layer = 50 * L    # Ensuring that on average, each of the L sites, receives 50 deposition events per layer 
T = num_layers * steps_per_layer    # Calculating the total number of deposition steps across all layers 
record_interval = 1000    # Setting the frequency for recording data (such as average height and roughness)

heights = np.zeros(L, dtype=int)    # Creating an array of zerros with length L to represent the initial height (zero particles) at each position along the substrate
snapshots = [np.zeros(L, dtype=int)]  # Initialising a list to store snapshots (copies) of the height profile 

times = []    # Setting up an empty list to store deposition steps at which data is recorded
avg_heights_over_time = []    # Setting up an empty list to store the average height of the substrate at each recorded time
roughness_over_time = []    # Setting up an empty list to store the standard deviation (roughness) at each recording 

t = 0    # Initialise a counter t to track the number of deposition steps 
for layer in range(num_layers):    # Beginning a loop to iterate over each deposition layer (3 iterations in total)
    for _ in range(steps_per_layer):    # Beginning a loop to iterate for the number of steps (10,000)
        i = np.random.randint(0, L)    # Randomly select an index i (between 0 and L-1), representing the position on the substrate 
        heights[i] += 1    # Increasing the height at the selected position by 1 (simulating the deposition of a particle)
        t += 1    # Increasing the time step counter by 1
        
        if t % record_interval == 0:    # Checking if the current step is a multiple of the record interval (1000, 2000, etc.) 
            times.append(t)    # Appending the current time step t to the time list 
            avg_heights_over_time.append(np.mean(heights))    # Computes the average height (mean of the heights array) and appends it to the list avg_heights_over_time
            roughness_over_time.append(np.std(heights))    # Calculates the standard deviation of the heights array (as a measure of roughness) and appends it to roughness_over_time
    
    snapshots.append(heights.copy())    # Saving a copy of the current heights array in the snapshot list, after completing a full layer of deposition (representing the state of the substrate after that layer)

plt.figure(figsize=(8, 5))    # Creating a new figure for plotting with a size of 8 inches by 5 inches
x = np.arange(L)    # Creating an array of positions from 0 to L-1 (x-axis values for the substrate positions)

colors = ["blue", "grey", "red"]    # Defining a list of colours to use for visualising each layer 

for i in range(num_layers, 0, -1):    # Looping backward from the  last snapshot to the first snapshot 
    plt.fill_between(x, 0, snapshots[i], color=colors[num_layers - i], step='pre')    # Using plt.fill_between to fill the area from y=0 up to the height profile in snapshots[i]

plt.xlim(0, L)    # Setting the limits of the x-axis from 0 to L 
plt.ylim(0, np.max(heights)*1.01)    # Setting the y-axis limits from 0 up to slightly above the maximum height
plt.xlabel('Position along the substrate')    # Setting labels 
plt.ylabel('Height')    # Setting labels 
plt.title("1D Random Ballistic Deposition")    # Adding a title 
plt.show()    # Displaying the plot 

times_arr = np.array(times)    # Converting the list into NumPy array for easier manipulation and analysis 
avg_heights_arr = np.array(avg_heights_over_time)    # Converting the list into NumPy array for easier manipulation and analysis 
roughness_arr = np.array(roughness_over_time)    # Converting the list into NumPy array for easier manipulation and analysis 

reg_avg = stats.linregress(times_arr, avg_heights_arr)    # Performing a linear regression on the average height versus time data (reg_avg holds regression parameters such as slope, intercept, r-value, etc.)

plt.figure(figsize=(8, 5))    # Creating a new figure for the average height plot
plt.plot(times_arr, avg_heights_arr, 'b.', label="Average Height Data")    # Plotting the average height data as blue dots ('b.') with a label for the legend
plt.plot(times_arr, reg_avg.slope * times_arr + reg_avg.intercept, 'r-',    
         label=f"Fit: slope={reg_avg.slope:.4f}, r={reg_avg.rvalue:.4f}")    # Adding labels
plt.xlabel("Time (deposition steps)")    # Setting labels 
plt.ylabel("Average Height")    # Setting labels 
plt.title("Average Height vs Time")    # Adding a title  
plt.legend()   # Adding a legend
plt.show()   # Displaying the plot 

log_times = np.log10(times_arr)    # Transforming the time data to logarithmic scale using base 10
log_roughness = np.log10(roughness_arr)    # Transforming the roughness data to logarithmic scale using base 10
reg_rough_log = stats.linregress(log_times, log_roughness)    # Performing linear regression on the log-transformed data

roughness_fit = 10 ** (reg_rough_log.slope * np.log10(times_arr) + reg_rough_log.intercept)    # Converting the regression line from log–log space back to the original scale

plt.figure(figsize=(8, 5))    # Creating a new figure for the roughness plot
plt.loglog(times_arr, roughness_arr, 'b.', label="Roughness Data")    # Plotting the fitted regression line (in red) on the log–log scale
plt.loglog(times_arr, roughness_fit, 'r-', 
           label=f"Fit: slope={reg_rough_log.slope:.4f}, r={reg_rough_log.rvalue:.4f}")    # Adding labels
plt.xlabel("Time (deposition steps)")     # Setting labels 
plt.ylabel("Roughness (Std. Deviation)")     # Setting labels 
plt.title("Roughness vs Time (Log-Log Scale)")    # Adding a title
plt.legend()    # Adding a legend
plt.show()    # Displaying the plot 