import numpy as np
import matplotlib.pyplot as plt

def random_step(prob):
    rnd = np.random.rand()
    return 1 if rnd <= prob else -1

prob_values = [0.3, 0.5, 0.8]
colors = ['gray', 'blue', 'red']
min_boundary = -10
max_boundary = 10
num_walkers = 1000
positions_range = range(21)  

plt.figure(figsize=(8, 5))
plt.grid(True)

for idx, p in enumerate(prob_values):
    mean_lifetimes = []
    x_positions = []
    
    for offset in positions_range:
        x_start = offset - 10  
        x_positions.append(x_start)
        times_to_escape = []
        
        for _ in range(num_walkers):
            position = x_start
            steps = 0
            while True:
                position += random_step(p)
                if position < min_boundary or position > max_boundary:
                    break
                steps += 1
            times_to_escape.append(steps)
        
        mean_lifetimes.append(np.mean(times_to_escape))
    
    plt.plot(
        x_positions,
        mean_lifetimes,
        'o--',
        markersize=3,
        color=colors[idx],
        label=f"p = {p}"
    )

plt.xlabel('Initial Position')
plt.ylabel('Mean Lifetime (steps)')
plt.title("Random Method: Mean Lifetime")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()