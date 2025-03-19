import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

def generate_lattice(L, p):
  
    return np.random.rand(L, L) < p

def flood_fill(lattice, labels, i, j, cluster_id):
  
    L = lattice.shape[0]
    stack = [(i, j)]
    labels[i, j] = cluster_id
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < L and 0 <= nc < L:
                if lattice[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = cluster_id
                    stack.append((nr, nc))
    return

def label_clusters(lattice):
  
    L = lattice.shape[0]
    labels = np.zeros((L, L), dtype=int)  
    cluster_id = 1  
    for i in range(L):
        for j in range(L):
            if lattice[i, j] and labels[i, j] == 0:
                flood_fill(lattice, labels, i, j, cluster_id)
                cluster_id += 1
    num_clusters = cluster_id - 1
    return labels, num_clusters

def detect_percolation(labels):

    L = labels.shape[0]
    left_labels = set(labels[:, 0]) - {0}
    right_labels = set(labels[:, -1]) - {0}
    percolating = left_labels & right_labels
    return percolating

def display_clusters(labels, percolating):
    L = labels.shape[0]
    max_label = labels.max()

    colors = ['black']  
    for label in range(1, max_label + 1):
        if label in percolating:
            colors.append('yellow')  
        else:
            colors.append(cm.nipy_spectral(label / max_label))
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_label + 1.5), cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(labels, cmap=cmap, norm=norm, origin='upper')
    plt.title("Yellow = Percolating")
    plt.axis('off')
    plt.show()

def main():
    L = int(input("Lattice size L: "))
    p = float(input("Probability p (between 0 and 1): "))
    
    lattice = generate_lattice(L, p)
    
    labels, num_clusters = label_clusters(lattice)
    print(f"Number of clusters found: {num_clusters}")
    
    percolating_clusters = detect_percolation(labels)
    if percolating_clusters:
        print("0")
        print("Percolating cluster(s):", percolating_clusters)
    else:
        print("0")
    
    display_clusters(labels, percolating_clusters)

if __name__ == "__main__":
    main()