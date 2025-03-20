import numpy as np
import matplotlib.pyplot as plt

def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]  
        i = parent[i]
    return i

def union(parent, rank, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i == root_j:
        return
    if rank[root_i] < rank[root_j]:
        parent[root_i] = root_j
    elif rank[root_i] > rank[root_j]:
        parent[root_j] = root_i
    else:
        parent[root_j] = root_i
        rank[root_i] += 1

def simulate_bond_percolation(L, p):
    N = L * L
    parent = np.arange(N)
    rank = np.zeros(N, dtype=int)
    
    def index(r, c):
        return r * L + c

    for r in range(L):
        for c in range(L):
            if c < L - 1:
                if np.random.rand() < p:  
                    union(parent, rank, index(r, c), index(r, c+1))
            if r < L - 1:
                if np.random.rand() < p:  
                    union(parent, rank, index(r, c), index(r+1, c))
    
    left_roots = set()
    for r in range(L):
        left_roots.add(find(parent, index(r, 0)))
    
    right_roots = set()
    for r in range(L):
        right_roots.add(find(parent, index(r, L-1)))
    
    return len(left_roots & right_roots) > 0

def run_simulation(L, p, num_runs=100):
    count = 0
    for _ in range(num_runs):
        if simulate_bond_percolation(L, p):
            count += 1
    return count / num_runs

def main():
    L_values = [10, 20, 40]
    delta_p = 0.05
    p_values = np.arange(0, 1 + delta_p, delta_p)
    results = {L: [] for L in L_values}
    
    for L in L_values:
        print(f"\nRunning simulations for L = {L}")
        for p in p_values:
            Q = run_simulation(L, p, num_runs=100)
            results[L].append(Q)
            print(f"  p = {p:.2f}, Q(p) = {Q:.2f}")
    
    plt.figure(figsize=(8, 6))
    for L in [10, 20, 40]:
        plt.plot(p_values, results[L], marker='o', label=f"L = {L}")
    
    plt.xlabel("Bond Probability p")
    plt.ylabel("Percolation Probability Q(p)")
    plt.title("Bond Percolation: Q(p) vs. p for Different Lattice Sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()