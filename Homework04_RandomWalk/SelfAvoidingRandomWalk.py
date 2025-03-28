import numpy as np
import matplotlib.pyplot as plt
import random

def single_self_avoiding_walk():
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    x, y = 0, 0
    visited = set()
    visited.add((x, y))
    steps = 0
    
    while True:
        possible_moves = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                possible_moves.append((dx, dy))
        
        if not possible_moves:
            return steps
        
        dx, dy = random.choice(possible_moves)
        x += dx
        y += dy
        visited.add((x, y))
        steps += 1

num_walks = 10000  
walk_lengths = []

for _ in range(num_walks):
    L = single_self_avoiding_walk()
    walk_lengths.append(L)

walk_lengths = np.array(walk_lengths)

plt.figure(figsize=(7,5))
plt.hist(walk_lengths, bins=50, density=True, edgecolor='black', alpha=0.7)
plt.xlabel("Walk Length Until Dead End")
plt.ylabel("Frequency (Normalised)")
plt.title("Distribution of Self-Avoiding Walk Lengths (2D Lattice)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("Number of walks:", num_walks)
print("Mean walk length:", walk_lengths.mean())
print("Max walk length:", walk_lengths.max())