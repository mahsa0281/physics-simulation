import numpy as np  # NumPy library for efficient numerical operations and array handling
import matplotlib.pyplot as plt  # Matplotlib’s pyplot module for creating plots and visualisations

def generate_pascals_triangle(n): # A function generating Pascal's Triangle up to n rows
    
    triangle = [] # Initialising an empty list to store the rows of Pascal's Triangle 
    row = [1] # Starting with the first row, which is always 1

    for i in range(n): # A loop running n times to generate n rows of Pascal's Triangle 
        triangle.append(row) # Saving the current row in the trianle list 
        new_row = [1] # Starts building the next row, the first element always being 1

        for j in range(len(row) - 1): # A loop calculating the middle values in the new row
            new_row.append(row[j] + row[j+1]) # Taking two adjacent numbers from the previous row, adding them, appening the sum
        new_row.append(1) # The last element of each row is always 1
        row = new_row # Updating row to be new_row, so the next iteration builds on it
    return triangle # Returning the full Pascal's Triangle a list of lists

num_rows = 64 # Defining the number of rows for the triangle 

triangle = generate_pascals_triangle(num_rows) # Calling the function to generate the triangle and storing it in triangle 

width = 2 * num_rows - 1 # Width: Each row has 2 * num_rows - 1 columns
image = np.ones((num_rows, width, 3))  # Creating a NumPy array to store pixel colors (height, width, RGB)


red = np.array([1, 0, 0]) # Defining colours in RGB format 
blue = np.array([0, 0, 1]) # Defining colours in RGB format

for r, row in enumerate(triangle): # Looping through each row of Pascal’s Triangle

    start = num_rows - r - 1  # Ensuring that each row starts at the correct position to be centered
    for i, value in enumerate(row): # Looping through each number in the row
        col = start + 2 * i  # Computing the column index for the i-th number in the row
        if value % 2 == 1: # Checking whether the number is odd or even
            image[r, col] = blue
        else:
            image[r, col] = red

plt.figure(figsize=(8, 8)) # Creating a figure with a size of 8x8
plt.imshow(image, interpolation='none')
plt.axis('off')
plt.title("Pascal's Triangle: Odd in Blue, Even in Red")
plt.show()
