import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

# =============== Side Deposition Simulation ===============
def side_deposition(L, H, N, record_interval):
    """
    Runs a 2D side-deposition simulation for a grid of size H x L,
    dropping N particles. Returns times (t), roughness array, and final grid.
    """
    deposit_grid = np.zeros((H, L), dtype=int)

    # Define neighbor offsets (below, below-left, below-right, left, right)
    neighbors = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)]

    def has_neighbor(grid, x, y):
        """Check if there's a deposited neighbor near (x,y)."""
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < L and 0 <= ny < H:
                if grid[ny, nx] != 0:
                    return True
        return False

    def get_surface_heights(grid):
        """
        For each column x, find the highest occupied cell (y-value).
        Returns a 1D array of surface heights for roughness calculation.
        """
        surface = np.zeros(L, dtype=int)
        for col in range(L):
            # Highest index y where deposit_grid[y, col] != 0
            occupied = np.where(grid[:, col] != 0)[0]
            if len(occupied) > 0:
                surface[col] = occupied[-1]  # topmost filled cell
            else:
                surface[col] = 0
        return surface

    t = 0
    times = []
    roughness_list = []

    # Main loop for depositing N particles
    for i in range(N):
        x = np.random.randint(0, L)
        y = H - 1

        # Move downward until hitting bottom or neighbor
        while True:
            if y == 0 or has_neighbor(deposit_grid, x, y):
                deposit_grid[y, x] = 1  # Mark deposited particle
                break
            else:
                y -= 1

        t += 1

        # Record roughness at specified intervals
        if t % record_interval == 0:
            times.append(t)
            surface = get_surface_heights(deposit_grid)
            roughness_list.append(np.std(surface))

    return np.array(times), np.array(roughness_list), deposit_grid


# =============== Single Run for Growth Exponent (beta) ===============
L = 200
H = 100
N = 3000
record_interval = 100

times, roughness, final_grid = side_deposition(L, H, N, record_interval)

# Log-log regression to find beta
log_times = np.log10(times)
log_roughness = np.log10(roughness)
reg_rough_log = stats.linregress(log_times, log_roughness)
beta = reg_rough_log.slope

print("Growth exponent β (from roughness vs. time):", beta)


# =============== Multiple L to Extract α and z ===============
L_values = [50, 100, 200, 400]
saturation_roughness = []
saturation_time = []

for L_val in L_values:
    # Keep height scaled similarly, or just use the same H
    H_val = 100
    N_val = 3000  # Adjust if needed for saturation
    times_val, roughness_val, grid_val = side_deposition(L_val, H_val, N_val, record_interval)

    # We'll assume the last recorded roughness is close to saturation
    W_sat = roughness_val[-1]
    t_sat = times_val[-1]

    saturation_roughness.append(W_sat)
    saturation_time.append(t_sat)

# Convert to NumPy arrays for log-log fits
L_arr = np.array(L_values, dtype=float)
W_sat_arr = np.array(saturation_roughness, dtype=float)
t_sat_arr = np.array(saturation_time, dtype=float)

# Roughness exponent alpha from W_sat ~ L^alpha
logL = np.log10(L_arr)
logW_sat = np.log10(W_sat_arr)
reg_alpha = stats.linregress(logL, logW_sat)
alpha = reg_alpha.slope

# Dynamic exponent z from t_s ~ L^z
logt_sat = np.log10(t_sat_arr)
reg_z = stats.linregress(logL, logt_sat)
z = reg_z.slope

print("Roughness exponent α (from W_s ∼ L^α):", alpha)
print("Dynamic exponent z (from t_s ∼ L^z):", z)
print("Scaling relation check (β * z):", beta * z)
