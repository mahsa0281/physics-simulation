import numpy as np
import matplotlib.pyplot as plt

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

for x in range(21):
    if x == 0:
        plt.plot([x], [lifetime_array[x]], marker='o', color='gray', markersize=2, label="p=0.3")
    else:
        plt.plot([x], [lifetime_array[x]], marker='o', color='gray', markersize=2)

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

for x in range(21):
    if x == 0:
        plt.plot([x], [lifetime_array[x]], marker='o', color='blue', markersize=2, label="p=0.5")
    else:
        plt.plot([x], [lifetime_array[x]], marker='o', color='blue', markersize=2)

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

for x in range(21):
    if x == 0:
        plt.plot([x], [lifetime_array[x]], marker='o', color='red', markersize=2, label="p=0.8")
    else:
        plt.plot([x], [lifetime_array[x]], marker='o', color='red', markersize=2)

plt.legend(loc="upper left")
plt.xlabel("Initial Site (x_0)")
plt.ylabel("Mean Lifetime")
plt.title("Enumeration Method: Mean Lifetime for p=0.3, 0.5, 0.8")
plt.show()