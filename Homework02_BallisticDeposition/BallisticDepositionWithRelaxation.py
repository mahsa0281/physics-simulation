import numpy as np   
import matplotlib.pyplot as plt    
from scipy import stats    
import random    

L = 200    
num_layers = 3   
steps_per_layer = 500 * L    
T = num_layers * steps_per_layer     
record_interval = 1000   

heights = np.zeros(L, dtype=int)   
snapshots = [np.zeros(L, dtype=int)]   

times = []     
avg_heights_over_time = []   
roughness_over_time = []   

t = 0   

for layer in range(num_layers):     
    for _ in range(steps_per_layer):    
        i = np.random.randint(0, L)   
        
        left_index = (i - 1) % L    
        right_index = (i + 1) % L 
        
        if heights[left_index] < heights[i] or heights[right_index] < heights[i]:    
            left_height = heights[left_index]     
            right_height = heights[right_index]  
            
            if left_height < right_height:   
                deposit_index = left_index   
            elif right_height < left_height: 
                deposit_index = right_index     
            else:   
                deposit_index = random.choice([left_index, right_index])     

            heights[deposit_index] += 1     
        else:     
            heights[i] += 1   
        
        t += 1    
        
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