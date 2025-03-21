import numpy as np   
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_lattice(L, p):
    lattice = np.random.rand(L, L) < p
    return lattice

def display_lattice(lattice, L, p):
    cmap = ListedColormap(["darkblue", "yellow"])
    
    plt.figure(figsize=(6, 6))
    plt.imshow(lattice, cmap=cmap, origin='upper')
    plt.title(f"Percolation Lattice: L = {L}, p = {p}")
    plt.axis('off')
    plt.show()

def has_spanning_cluster(lattice):
    L = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    
    def dfs(r, c):
        if c == L - 1:
            return True
        visited[r, c] = True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < L and 0 <= nc < L:
                if lattice[nr, nc] and not visited[nr, nc]:
                    if dfs(nr, nc):
                        return True
        return False

    for i in range(L):
        if lattice[i, 0] and not visited[i, 0]:
            if dfs(i, 0):
                return True
    return False

if __name__ == "__main__":
    L = int(input("Lattice size L: "))
    p = float(input("Probability P: "))

    lattice = generate_lattice(L, p)
    if has_spanning_cluster(lattice):
        print("1")
    else:
        print("0")
    
    display_lattice(lattice, L, p)