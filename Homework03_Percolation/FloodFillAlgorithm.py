import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

def propagate_label(labels, i, j, new_label):
    L = labels.shape[0]
    stack = [(i, j)]
    while stack:
        r, c = stack.pop()
        if labels[r, c] > new_label:
            labels[r, c] = new_label
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < L and 0 <= nc < L:
                if labels[nr, nc] != 0 and labels[nr, nc] > new_label:
                    labels[nr, nc] = new_label
                    stack.append((nr, nc))

def color_lattice(L, p):
    int_max = 10**6  
    labels = np.zeros((L, L), dtype=int)
    
    for i in range(L):
        labels[i, 0] = 1
        
    for i in range(L):
        labels[i, L-1] = int_max
        
    next_label = 2  

    for i in range(L):
        for j in range(1, L-1):
            if labels[i, j] == 0:
                if np.random.rand() < p:
                    neighbor_vals = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = i + dr, j + dc
                        if 0 <= nr < L and 0 <= nc < L:
                            if labels[nr, nc] != 0:
                                neighbor_vals.append(labels[nr, nc])
                    if not neighbor_vals:
                        labels[i, j] = next_label
                        next_label += 1
                    elif len(neighbor_vals) == 1:
                        labels[i, j] = neighbor_vals[0]
                    else:
                        min_label = min(neighbor_vals)
                        labels[i, j] = min_label
                        propagate_label(labels, i, j, min_label)
    return labels

def detect_percolation(labels):
    L = labels.shape[0]
    return any(labels[i, L-1] == 1 for i in range(L))

def display_lattice(labels, L, p):
  
    max_label = labels.max()
    colors = ['black']  
    for label in range(1, max_label + 1):
        if label == 1:
            colors.append('yellow')
        else:
            colors.append(cm.nipy_spectral(label / max_label))
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_label + 1.5), cmap.N)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(labels, cmap=cmap, norm=norm, origin='upper')
    plt.title(f"Lattice Coloring (L = {L}, p = {p})")
    plt.axis('off')
    plt.show()

def main():
    L = int(input("Lattice size L: "))
    p = float(input("Probability p (between 0 and 1): "))
    
    start_time = time.time()
    labels = color_lattice(L, p)
    percolates = detect_percolation(labels)
    
    if percolates:
        print("1")
    else:
        print("0")
    
    end_time = time.time()
    print(f"Elapsed time for computation: {end_time - start_time:.4f} seconds")
    
    display_lattice(labels, L, p)

if __name__ == "__main__":
    main()