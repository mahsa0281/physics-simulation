import numpy as np
import matplotlib.pyplot as plt

MOVES = [(1,0), (-1,0), (0,1), (0,-1)]

def count_saws(N):
    def dfs(x, y, steps, visited):
        if steps == N:
            return 1
        count = 0
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                count += dfs(nx, ny, steps + 1, visited)
                visited.remove((nx, ny))
        return count
    
    visited = set()
    visited.add((0, 0))  
    return dfs(0, 0, 0, visited)

def main():
    max_N = 15  
    
    saw_counts = []
    ratio_values = []
    N_values = range(1, max_N + 1)
    
    for N in N_values:
        num_saws = count_saws(N)
        saw_counts.append(num_saws)
        num_free = 4**N
        ratio = num_saws / num_free
        
        ratio_values.append(ratio)
        
        print(f"N={N}, #SAWs={num_saws}, ratio(SAWs/4^N)={ratio:.6g}")
    
    N_arr = np.array(list(N_values))
    saw_arr = np.array(saw_counts, dtype=float)
    ratio_arr = np.array(ratio_values, dtype=float)
    
    plt.figure(figsize=(7,5))
    plt.plot(N_arr, saw_arr, 'o-', label="SAWs Count")
    plt.xlabel("Walk Length N")
    plt.ylabel("Number of Self-Avoiding Walks")
    plt.title("Count of Self-Avoiding Walks vs. N (2D Lattice)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(7,5))
    plt.plot(N_arr, ratio_arr, 'o-', color='red', label=r"SAWs / $4^N$")
    plt.xlabel("Walk Length N")
    plt.ylabel(r"Ratio of SAWs to $4^N$")
    plt.title("Ratio of Self-Avoiding Walks to Free Walks vs. N")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()