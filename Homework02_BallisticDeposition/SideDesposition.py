import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

L = 200      # Width of the 1D substrate (x-direction)
H = 100      # Maximum height for the grid (y-direction)
N = 3000     # Number of falling particles

deposit_grid = np.zeros((H, L), dtype=int)

neighbors = [
    (0, -1),    # below
    (-1, -1),   # below-left
    (1, -1),    # below-right
    (-1, 0),    # left
    (1, 0)      # right
]

def has_neighbor(grid, x, y):
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < L and 0 <= ny < H:
            if grid[ny, nx] != 0:  
                return True
    return False


for i in range(N):
    batch_id = i // 1000 + 1  

    x = np.random.randint(0, L)
    y = H - 1
    
    while True:
        if y == 0 or has_neighbor(deposit_grid, x, y):
            deposit_grid[y, x] = batch_id
            break
        else:
            y -= 1


my_cmap = mcolors.ListedColormap(["white", "red", "gray", "blue"])

plt.figure(figsize=(8, 5))
plt.imshow(
    deposit_grid,
    origin='lower',
    extent=(0, L, 0, H),
    cmap=my_cmap,
    vmin=0,  
    vmax=3   
)
plt.xlabel('X (horizontal sites)')
plt.ylabel('Y (height)')
plt.title('Side Deposition with 3 Distinct Colors by Batch')
plt.show()