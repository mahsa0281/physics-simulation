import numpy as np
import matplotlib.pyplot as plt
import random

num_trials = 1000

sample_sizes = np.array([5, 10, 100, 1000])

for sample_size in sample_sizes:
    sum_results = np.array([])
    
    for _ in range(num_trials):
        sample = np.array([])
        for _ in range(sample_size):
            sample = np.append(sample, random.randint(0, 9))
        
        sample_sum = np.sum(sample)
        sum_results = np.append(sum_results, sample_sum)
    
    num_bins = (-0.00004871 * sample_size**2) + (0.0591463 * sample_size) + 9.56376
    
    hist_counts, bin_edges, _ = plt.hist(sum_results, bins=int(num_bins) + 1)
    
    mean_val = np.average(sum_results)
    std_val = np.std(sum_results)
    
    x_values = np.linspace(min(sum_results), max(sum_results), 100)
    gaussian_curve = np.exp(-0.5 * ((x_values - mean_val) / std_val)**2) * max(hist_counts)
    plt.plot(x_values, gaussian_curve, 'r')
    
    plt.xlabel("Sum of Random Digits")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Sums for Sample Size = {sample_size} ({num_trials} trials)")
    plt.show()