import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import imageio

N = 60
rho = 0.9
Bm = 0.6
max_steps = 50 
save_dir = "frames"
os.makedirs(save_dir, exist_ok=True)

def initialize_grid(N, rho, group_ratio=0.5):
    total_cells = N * N
    num_occupied = int(total_cells * rho)
    num_group1 = int(num_occupied * group_ratio)
    num_group2 = num_occupied - num_group1
    num_empty = total_cells - num_occupied

    cells = np.array([0]*num_empty + [1]*num_group1 + [2]*num_group2)
    np.random.shuffle(cells)
    return cells.reshape((N, N))

def is_happy(grid, x, y, Bm):
    agent = grid[x, y]
    if agent == 0:
        return True
    same, total = 0, 0
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                neighbor = grid[nx, ny]
                if neighbor != 0:
                    total += 1
                    if neighbor == agent:
                        same += 1
    if total == 0:
        return True
    return (same / total) >= Bm

def move_unhappy_agents(grid, Bm):
    new_grid = grid.copy()
    unhappy = [(x, y) for x in range(N) for y in range(N)
               if new_grid[x, y] != 0 and not is_happy(new_grid, x, y, Bm)]
    empties = list(zip(*np.where(new_grid == 0)))

    np.random.shuffle(unhappy)
    np.random.shuffle(empties)

    moves = min(len(unhappy), len(empties))
    for i in range(moves):
        x_old, y_old = unhappy[i]
        x_new, y_new = empties[i]
        new_grid[x_new, y_new] = new_grid[x_old, y_old]
        new_grid[x_old, y_old] = 0

    return new_grid, len(unhappy)

def save_frame(grid, step):
    cmap = mcolors.ListedColormap(['white', 'blue', 'red'])
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap)
    plt.title(f"Step {step}")
    plt.axis('off')
    plt.savefig(f"{save_dir}/frame_{step:03d}.png")
    plt.close()

grid = initialize_grid(N, rho)
for step_num in range(max_steps):
    save_frame(grid, step_num)
    grid, num_unhappy = move_unhappy_agents(grid, Bm)
    print(f"Step {step_num}: {num_unhappy} unhappy agents")
    if num_unhappy == 0:
        break

images = [imageio.imread(f"{save_dir}/frame_{i:03d}.png") for i in range(step_num + 1)]
imageio.mimsave("schelling_simulation.gif", images, fps=2)
print("GIF saved as schelling_simulation.gif")