import numpy as np
import matplotlib.pyplot as plt

def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i == root_j:
        return
    
    if rank[root_i] < rank[root_j]:
        parent[root_i] = root_j
        size[root_j] += size[root_i]
        size[root_i] = 0
        sum_x[root_j] += sum_x[root_i]
        sum_y[root_j] += sum_y[root_i]
        sum_x2[root_j] += sum_x2[root_i]
        sum_y2[root_j] += sum_y2[root_i]
        sum_x[root_i] = 0
        sum_y[root_i] = 0
        sum_x2[root_i] = 0
        sum_y2[root_i] = 0
    elif rank[root_i] > rank[root_j]:
        parent[root_j] = root_i
        size[root_i] += size[root_j]
        size[root_j] = 0
        sum_x[root_i] += sum_x[root_j]
        sum_y[root_i] += sum_y[root_j]
        sum_x2[root_i] += sum_x2[root_j]
        sum_y2[root_i] += sum_y2[root_j]
        sum_x[root_j] = 0
        sum_y[root_j] = 0
        sum_x2[root_j] = 0
        sum_y2[root_j] = 0
    else:
        parent[root_j] = root_i
        rank[root_i] += 1
        size[root_i] += size[root_j]
        size[root_j] = 0
        sum_x[root_i] += sum_x[root_j]
        sum_y[root_i] += sum_y[root_j]
        sum_x2[root_i] += sum_x2[root_j]
        sum_y2[root_i] += sum_y2[root_j]
        sum_x[root_j] = 0
        sum_y[root_j] = 0
        sum_x2[root_j] = 0
        sum_y2[root_j] = 0

def simulate_bond_percolation(L, p):
    N = L * L
    parent = np.arange(N)
    rank = np.zeros(N, dtype=int)
    size = np.ones(N, dtype=int)  
    sum_x  = np.zeros(N, dtype=float)
    sum_y  = np.zeros(N, dtype=float)
    sum_x2 = np.zeros(N, dtype=float)
    sum_y2 = np.zeros(N, dtype=float)
    
    def index(r, c):
        return r * L + c
    
    for r in range(L):
        for c in range(L):
            idx = index(r, c)
            sum_x[idx]  = c
            sum_y[idx]  = r
            sum_x2[idx] = c*c
            sum_y2[idx] = r*r

    for r in range(L):
        for c in range(L):
            if c < L - 1:
                if np.random.rand() < p:  
                    union(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, index(r, c), index(r, c+1))
            if r < L - 1:
                if np.random.rand() < p:  
                    union(parent, rank, size, sum_x, sum_y, sum_x2, sum_y2, index(r, c), index(r+1, c))
    
    left_roots  = set()
    right_roots = set()
    for r in range(L):
        left_roots.add(find(parent, index(r, 0)))
        right_roots.add(find(parent, index(r, L-1)))
    spanning_roots = left_roots & right_roots
    
    Rg_values = []
    
    for root in range(N):
        if parent[root] == root and size[root] > 0:
            if root in spanning_roots:
                continue
            
            S = size[root]
            if S <= 1:
                Rg_values.append(0.0)
                continue
            
            sumx  = sum_x[root]
            sumy  = sum_y[root]
            sumx2 = sum_x2[root]
            sumy2 = sum_y2[root]
            x_c = sumx / S
            y_c = sumy / S
            mean_r2 = (sumx2 + sumy2) / S
            Rg2 = mean_r2 - x_c*x_c - y_c*y_c
            Rg = np.sqrt(Rg2) if Rg2 > 0 else 0.0
            
            Rg_values.append(Rg)
    
    if len(Rg_values) == 0:
        return 0.0
        xi = np.mean(Rg_values)
    return xi

def run_simulation(L, p, num_runs=100):
    xi_sum = 0.0
    for _ in range(num_runs):
        xi_sum += simulate_bond_percolation(L, p)
    return xi_sum / num_runs

def main():
    L_values = [10, 20, 40, 80]
    
    delta_p = 0.05
    p_values = np.arange(0, 1 + delta_p, delta_p)
    
    results = {L: [] for L in L_values}
    
    for L in L_values:
        print(f"\nRunning correlation-length simulations for L = {L}")
        for p in p_values:
            xi_avg = run_simulation(L, p, num_runs=50) 
            results[L].append(xi_avg)
            print(f"  p = {p:.2f}, xi(p) = {xi_avg:.3f}")
    
    plt.figure(figsize=(8,6))
    for L in [10, 20, 40, 80]:
        plt.plot(p_values, results[L], marker='o', label=f"L = {L}")
    
    plt.xlabel("Bond Probability p")
    plt.ylabel(r"Average Correlation Length $\xi(p)$")
    plt.title("Correlation Length vs. p (Bond Percolation)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()