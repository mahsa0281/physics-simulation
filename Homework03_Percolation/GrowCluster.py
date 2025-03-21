import numpy as np
import matplotlib.pyplot as plt
import random

def grow_cluster(L, p, seed=None):
    lattice = np.zeros((L, L), dtype=int)
    
    if seed is None:
        r = np.random.randint(0, L)
        c = np.random.randint(0, L)
    else:
        r, c = seed
    
    lattice[r, c] = 1
    
    from collections import deque
    queue = deque()
    queue.append((r, c))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < L and 0 <= nc < L:
                if lattice[nr, nc] == 0:
                    if random.random() < p:
                        lattice[nr, nc] = 1
                        queue.append((nr, nc))
                    else:
                        lattice[nr, nc] = -1
    
    occupied_coords = np.argwhere(lattice == 1)
    S = len(occupied_coords) 
    
    if S <= 1:
        return S, 0.0
    
    x_coords = occupied_coords[:, 1]  
    y_coords = occupied_coords[:, 0]  
    sum_x  = x_coords.sum()
    sum_y  = y_coords.sum()
    sum_x2 = (x_coords**2).sum()
    sum_y2 = (y_coords**2).sum()
    x_c = sum_x / S
    y_c = sum_y / S
    mean_r2 = (sum_x2 + sum_y2) / S
    Rg2 = mean_r2 - x_c**2 - y_c**2
    Rg = np.sqrt(Rg2) if Rg2 > 0 else 0.0
    
    return S, Rg

def main():
    L = 200
    p_values = [0.50, 0.55, 0.59]
    num_clusters = 50
    data = {p: [] for p in p_values}
    
    for p in p_values:
        for _ in range(num_clusters):
            S, Rg = grow_cluster(L, p)
            data[p].append((S, Rg))

    plt.figure(figsize=(8, 6))
    S_all = []
    Rg_all = []

    for p in p_values:
        S_vals = np.array([pair[0] for pair in data[p]])
        Rg_vals = np.array([pair[1] for pair in data[p]])
        mask = (S_vals > 1) & (Rg_vals > 0)
        S_vals = S_vals[mask]
        Rg_vals = Rg_vals[mask]
        
        if len(S_vals) > 0:
            S_all.extend(S_vals)
            Rg_all.extend(Rg_vals)
            plt.scatter(Rg_vals, S_vals, alpha=0.6, label=f"p={p}")

    S_all = np.array(S_all)
    Rg_all = np.array(Rg_all)
    logRg_all = np.log10(Rg_all)
    logS_all = np.log10(S_all)
    slope, intercept = np.polyfit(logRg_all, logS_all, 1)
    a = 10**intercept
    Rg_fit = np.linspace(Rg_all.min(), Rg_all.max(), 100)
    S_fit = a * (Rg_fit**slope)

    plt.plot(Rg_fit, S_fit, 'k--', 
             label=f"Global Fit: S = {a:.2f} * Rg^{slope:.2f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Radius of Gyration Rg (log scale)")
    plt.ylabel("Cluster Size S (log scale)")
    plt.title("Site Percolation Cluster Growth (Single-Cluster Algorithm)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()