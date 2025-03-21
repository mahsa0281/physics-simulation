import numpy as np
import matplotlib.pyplot as plt

def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(parent, rank, size, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i == root_j:
        return
    if rank[root_i] < rank[root_j]:
        parent[root_i] = root_j
        size[root_j] += size[root_i]   
        size[root_i] = 0              
    elif rank[root_i] > rank[root_j]:
        parent[root_j] = root_i
        size[root_i] += size[root_j]
        size[root_j] = 0
    else:
        parent[root_j] = root_i
        rank[root_i] += 1
        size[root_i] += size[root_j]
        size[root_j] = 0

def simulate_bond_percolation(L, p):
    N = L * L  
    parent = np.arange(N)
    rank = np.zeros(N, dtype=int)
    size = np.ones(N, dtype=int)  
    
    def index(r, c):
        return r * L + c

    for r in range(L):
        for c in range(L):
            if c < L - 1:
                if np.random.rand() < p: 
                    union(parent, rank, size, index(r, c), index(r, c+1))
            if r < L - 1:
                if np.random.rand() < p: 
                    union(parent, rank, size, index(r, c), index(r+1, c))
    
    left_roots = set()
    right_roots = set()
    for r in range(L):
        left_roots.add(find(parent, index(r, 0)))
        right_roots.add(find(parent, index(r, L-1)))
    
    spanning_roots = left_roots & right_roots
    
    if not spanning_roots:
        return 0.0
    
    total_spanning_size = 0
    for root in spanning_roots:
        total_spanning_size += size[root]
        Q_infty = total_spanning_size / float(N)
    return Q_infty

def run_simulation(L, p, num_runs=100):
    Q_sum = 0.0
    for _ in range(num_runs):
        Q_sum += simulate_bond_percolation(L, p)
    return Q_sum / num_runs

def main():
    L_values = [10, 20, 160]
    
    delta_p = 0.05
    p_values = np.arange(0, 1 + delta_p, delta_p)
    
    results = {L: [] for L in L_values}
    
    for L in L_values:
        print(f"\nRunning Q_infty simulations for L = {L}")
        for p in p_values:
            Q_avg = run_simulation(L, p, num_runs=100)
            results[L].append(Q_avg)
            print(f"  p = {p:.2f}, Q_infty(p) = {Q_avg:.3f}")
    
    plt.figure(figsize=(8, 6))
    for L in [10, 20, 160]:
        plt.plot(p_values, results[L], marker='o', label=f"L = {L}")
    
    plt.xlabel("Bond Probability p")
    plt.ylabel(r"Average $Q_{\infty}(p)$")
    plt.title("Fraction of Sites in Spanning Cluster vs. p (Bond Percolation)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()