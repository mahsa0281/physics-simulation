import numpy as np
import matplotlib.pyplot as plt


L = 200        # Width
H = 100        # Height
N = 9000       # Number of falling particles

grid = np.zeros((H, L), dtype=int)


seed_x = L // 2
seed_y = 0
grid[seed_y, seed_x] = 1


neighbors = [
    (0, 0),    # the cell itself
    (0, -1),   # below
    (-1, -1),  # below-left
    (1, -1),   # below-right
    (-1, 0),   # left
    (1, 0)     # right
]

def has_occupied_neighbor(g, x, y):
    """
    Check if (x, y) or its neighbors is occupied.
    """
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < L and 0 <= ny < H:
            if g[ny, nx] == 1:
                return True
    return False


for _ in range(N):
    x = np.random.randint(0, L)
    y = H - 1
    
    while y >= 0:
        if has_occupied_neighbor(grid, x, y):
            grid[y, x] = 1
            break
        else:
            y -= 1

plt.figure(figsize=(8, 5))
plt.imshow(grid, origin='lower', cmap='binary', extent=(0, L, 0, H))
plt.title("Growth from a Single Seed Point")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()