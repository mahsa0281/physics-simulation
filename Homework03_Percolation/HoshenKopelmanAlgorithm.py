import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

def hoshen_kopelman(L, p):
    occupied = np.zeros((L, L), dtype=bool)
    
    for i in range(L):
        occupied[i, 0] = True
    
    for i in range(L):
        for j in range(1, L):
            if np.random.rand() < p:
                occupied[i, j] = True

    labels = np.zeros((L, L), dtype=int)
    
    parent = {} 
    size = {}    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return root_x
        rep = min(root_x, root_y)
        other = max(root_x, root_y)
        parent[other] = rep
        size[rep] += size[other]
        return rep
    
    next_label = 2  
    
    for j in range(L):
        for i in range(L):
            if not occupied[i, j]:
                continue
            if j == 0:
                labels[i, j] = 1
                if 1 not in parent:
                    parent[1] = 1
                    size[1] = 0
                size[1] += 1
            else:
                top_label = labels[i-1, j] if i > 0 else 0
                left_label = labels[i, j-1]
                
                if top_label == 0 and left_label == 0:
                    labels[i, j] = next_label
                    parent[next_label] = next_label
                    size[next_label] = 1
                    next_label += 1
                elif top_label != 0 and left_label == 0:
                    root = find(top_label)
                    labels[i, j] = root
                    size[root] += 1
                elif top_label == 0 and left_label != 0:
                    root = find(left_label)
                    labels[i, j] = root
                    size[root] += 1
                else:
                    root_top = find(top_label)
                    root_left = find(left_label)
                    if root_top == root_left:
                        labels[i, j] = root_top
                        size[root_top] += 1
                    else:
                        new_root = min(root_top, root_left)
                        labels[i, j] = new_root
                        union(new_root, max(root_top, root_left))
                        size[new_root] += 1  
    
    for i in range(L):
        for j in range(L):
            if labels[i, j] != 0:
                labels[i, j] = find(labels[i, j])
    
    return labels, occupied

def detect_percolation_HK(labels):
    L = labels.shape[0]
    for i in range(L):
        if labels[i, L-1] == 1:
            return True
    return False

def display_clusters(labels, L, p):
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
    plt.title(f"Hoshen–Kopelman Clusters (L = {L}, p = {p})")
    plt.axis('off')
    plt.show()

def main():
    L = int(input("Lattice size L: "))
    p = float(input("Probability p (between 0 and 1): "))
    
    start_time = time.time()
    labels, occupied = hoshen_kopelman(L, p)
    percolates = detect_percolation_HK(labels)
    
    if percolates:
        print("1")
    else:
        print("0")
    
    end_time = time.time()
    print(f"Elapsed time for computation: {end_time - start_time:.4f} seconds")
    
    display_clusters(labels, L, p)

if __name__ == "__main__":
    main()