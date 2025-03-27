import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.grid(True)

prob_step = 0.3
q_step = 1 - prob_step
lifetime_array = np.zeros(21)

for start_site in range(21):
    pdist = np.array([
        [0.0 for _ in range(21)],
        [0.0 for _ in range(21)]
    ])
    pdist[0][start_site] = 1.0
    
    time_step = 1
    accumulated_time = 0.0
    
    while True:
        pdist = np.append(pdist, [[0.0 for _ in range(21)]], axis=0)
    
        for site in range(21):
            if 1 < site < 19:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site - 1] * prob_step +
                    pdist[time_step - 1][site + 1] * q_step
                )
            elif site == 1:
                pdist[time_step][site] = pdist[time_step - 1][site + 1] * q_step
            elif site == 19:
                pdist[time_step][site] = pdist[time_step - 1][site - 1] * prob_step
            elif site == 20:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site] +
                    pdist[time_step - 1][site - 1] * prob_step
                )
            elif site == 0:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site] +
                    pdist[time_step - 1][site + 1] * q_step
                )
        
        newly_absorbed = (
            pdist[time_step][0] + pdist[time_step][20]
            - (pdist[time_step - 1][0] + pdist[time_step - 1][20])
        )
        accumulated_time += newly_absorbed * time_step
        
        if (pdist[time_step][0] + pdist[time_step][20]) > 0.9999:
            lifetime_array[start_site] = accumulated_time
            break
        
        time_step += 1

plt.plot(range(21), lifetime_array, 'o--', markersize=3, color='gray', label="p = 0.3")

prob_step = 0.5
q_step = 1 - prob_step
lifetime_array = np.zeros(21)

for start_site in range(21):
    pdist = np.array([
        [0.0 for _ in range(21)],
        [0.0 for _ in range(21)]
    ])
    pdist[0][start_site] = 1.0
    
    time_step = 1
    accumulated_time = 0.0
    
    while True:
        pdist = np.append(pdist, [[0.0 for _ in range(21)]], axis=0)
    
        for site in range(21):
            if 1 < site < 19:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site - 1] * prob_step +
                    pdist[time_step - 1][site + 1] * q_step
                )
            elif site == 1:
                pdist[time_step][site] = pdist[time_step - 1][site + 1] * q_step
            elif site == 19:
                pdist[time_step][site] = pdist[time_step - 1][site - 1] * prob_step
            elif site == 20:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site] +
                    pdist[time_step - 1][site - 1] * prob_step
                )
            elif site == 0:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site] +
                    pdist[time_step - 1][site + 1] * q_step
                )
        
        newly_absorbed = (
            pdist[time_step][0] + pdist[time_step][20]
            - (pdist[time_step - 1][0] + pdist[time_step - 1][20])
        )
        accumulated_time += newly_absorbed * time_step
        
        if (pdist[time_step][0] + pdist[time_step][20]) > 0.9999:
            lifetime_array[start_site] = accumulated_time
            break
        
        time_step += 1

plt.plot(range(21), lifetime_array, 'o--', markersize=3, color='blue', label="p = 0.5")

prob_step = 0.8
q_step = 1 - prob_step
lifetime_array = np.zeros(21)

for start_site in range(21):
    pdist = np.array([
        [0.0 for _ in range(21)],
        [0.0 for _ in range(21)]
    ])
    pdist[0][start_site] = 1.0
    
    time_step = 1
    accumulated_time = 0.0
    
    while True:
        pdist = np.append(pdist, [[0.0 for _ in range(21)]], axis=0)
    
        for site in range(21):
            if 1 < site < 19:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site - 1] * prob_step +
                    pdist[time_step - 1][site + 1] * q_step
                )
            elif site == 1:
                pdist[time_step][site] = pdist[time_step - 1][site + 1] * q_step
            elif site == 19:
                pdist[time_step][site] = pdist[time_step - 1][site - 1] * prob_step
            elif site == 20:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site] +
                    pdist[time_step - 1][site - 1] * prob_step
                )
            elif site == 0:
                pdist[time_step][site] = (
                    pdist[time_step - 1][site] +
                    pdist[time_step - 1][site + 1] * q_step
                )
        
        newly_absorbed = (
            pdist[time_step][0] + pdist[time_step][20]
            - (pdist[time_step - 1][0] + pdist[time_step - 1][20])
        )
        accumulated_time += newly_absorbed * time_step
        
        if (pdist[time_step][0] + pdist[time_step][20]) > 0.9999:
            lifetime_array[start_site] = accumulated_time
            break
        
        time_step += 1

plt.plot(range(21), lifetime_array, 'o--', markersize=3, color='red', label="p = 0.8")

plt.xlabel("Initial Site (x₀)")
plt.ylabel("Mean Lifetime")
plt.title("Enumeration Method: Mean Lifetime")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
