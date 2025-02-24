import numpy as np   # Used for efficient numerical computations 
import matplotlib.pyplot as plt   # Used for visualising 

def julia_set(width, height, c, xlim=(-1.7, 1.7), ylim=(-1.7,1.7), max_iter=350):   # Defining a function that generates the Julia Set

    x = np.linspace(xlim[0], xlim[1], width)   # Generating evenly spaced points between the given limits 
    y = np.linspace(ylim[0], ylim[1], height)
    X, Y = np.meshgrid(x, y)   # Creating a 2D grid of points
    Z = X + 1j * Y   # Converting the grid into complex numbers 

    iterations = np.zeros(Z.shape, dtype=int)   # Creating a 2D array of zeros 

    for i in range(max_iter):   # Looping for max_iter times 
        mask = np.abs(Z) < 2    # Checking which points haven't escaped (|Z| < 2)
        Z[mask] = Z[mask] ** 2 + c    # Updating only those points, applying the function 
        iterations[mask] += 1    # Increasing the iteration count for those points 

    return iterations

width, height = 1000, 1000    # Defining the image resolution 
max_iter = 350    # Maximum iterations to track divergence 
 
c_values = [(-0.8 + 0.16j), (-0.4 + 0.6j), (0.381 + 0.321j), (-0.711 - 0.3002j)]    # Complex values defining the Julia Sets 

titles = [r"$c = -0.8 + 0.16i$", r"$c = -0.4 + 0.6i$", r"$c = 0.381 + 0.321i$", r"$c = -0.711 - 0.3002i$"] # Labeling the images

for c, title in zip(c_values, titles):   # Looping through each c value (and it's corresponding title) and generating Julia Sets 
    julia = julia_set(width, height, c, max_iter=max_iter)    # Calling the julia_set() function to compute the fractal
    
    plt.figure(figsize=(6, 6))   # Creating a figure 
    plt.imshow(julia, cmap='inferno', extent=[-2, 2, -2, 2])   # Displaying the Julia set as an image 
    plt.colorbar(label="Iterations")    # Adding a colour bar showing iteration counts 
    plt.title(f"Julia Set for {title}")    # Adding labels for clarity 
    plt.xlabel("Re(z)")   # Adding labels for clarity 
    plt.ylabel("Im(z)")   # Adding labels for clarity 
    
    filename = f"julia_set_{c.real}_{c.imag}.png".replace(".", "_")   # Creating a filename based on the real and the imaginary parts of c
    plt.savefig(filename, dpi=300, bbox_inches='tight')   # Saving the image with high resolution 
    plt.show()   # Displaying the image 

    print(f"Saved: {filename}")    # Printing the saved filename 