import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as colors

def within_bounding_box(x, y, min_x, max_x, min_y, max_y, margin):
    if x < min_x - margin or x > max_x + margin:
        return False
    if y < min_y - margin or y > max_y + margin:
        return False
    return True

def neighbors4(x, y):
    return [(x,   y+1),
            (x,   y-1),
            (x-1, y  ),
            (x+1, y  )]

GRID_W, GRID_H = 400, 400
cluster = np.zeros((GRID_H, GRID_W), dtype=int)  

seed_y = 50
seed_x_start = 100
seed_length = 200
for sx in range(seed_x_start, seed_x_start + seed_length):
    cluster[seed_y, sx] = 1  

min_x, max_x = seed_x_start, seed_x_start + seed_length - 1
min_y, max_y = seed_y, seed_y

NUM_PARTICLES = 15000 
MARGIN = 50           

current_id = 2 

for _ in range(NUM_PARTICLES):
    start_x = random.randint(min_x, max_x)
    start_y = max_y + 20
    
    x, y = start_x, start_y
    
    while True:
        if not within_bounding_box(x, y, min_x, max_x, min_y, max_y, MARGIN):
            break
        
        stuck = False
        for nx, ny in neighbors4(x, y):
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if cluster[ny, nx] != 0:
                    stuck = True
                    break
        
        if stuck:
            cluster[y, x] = current_id
            current_id += 1
            
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
            
            break  
        
        step = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        x += step[0]
        y += step[1]

masked_cluster = np.ma.masked_where(cluster == 0, cluster)

cmap = plt.cm.rainbow

vmin = 1
vmax = masked_cluster.max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
ax.set_facecolor('white')

im = ax.imshow(masked_cluster, origin='lower',
               cmap=cmap, norm=norm, interpolation='none')

ax.set_title("2D DLA (Linear Seed)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()