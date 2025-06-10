import numpy as np
import matplotlib.pyplot as plt

N = 60
Bm_fixed = 0.4
rho_values = np.linspace(0.6, 0.95, 8) 
runs_per_rho = 5

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
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
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

def run_schelling(N, Bm, rho, max_steps=100):
    grid = initialize_grid(N, rho)
    moves = []
    for _ in range(max_steps):
        grid, num_unhappy = move_unhappy_agents(grid, Bm)
        moves.append(num_unhappy)
        if num_unhappy == 0:
            break
    return grid, moves

average_moves = []

for rho in rho_values:
    total = 0
    for _ in range(runs_per_rho):
        _, moves = run_schelling(N, Bm_fixed, rho)
        total += sum(moves)
    avg = total / runs_per_rho
    average_moves.append(avg)
    print(f"rho={rho:.2f} → avg moves = {avg:.1f}")

plt.figure(figsize=(8, 5))
plt.plot(rho_values, average_moves, marker='s')
plt.title(f"Avg Number of Moves vs Population Density ρ (Bm = {Bm_fixed})")
plt.xlabel("Population Density ρ")
plt.ylabel("Average Total Moves")
plt.grid(True)
plt.tight_layout()
plt.show()