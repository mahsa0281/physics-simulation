import numpy as np  
import matplotlib.pyplot as plt  
from scipy import stats  

L = 200  
num_layers = 3   
steps_per_layer = 50 * L   
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
        heights[i] += 1  
        t += 1    
        
        if t % record_interval == 0: 
            times.append(t)    
            avg_heights_over_time.append(np.mean(heights))   
            roughness_over_time.append(np.std(heights))    
    
    snapshots.append(heights.copy())    

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