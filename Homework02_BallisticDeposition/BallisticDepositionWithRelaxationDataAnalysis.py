import numpy as np
from scipy import stats
import random

def simulate_ballistic_deposition(L, num_layers, steps_per_layer, record_interval):
    heights = np.zeros(L, dtype=int)
    times = []
    roughness_over_time = []
    t = 0  
    for layer in range(num_layers):
        for _ in range(steps_per_layer):
            i = np.random.randint(0, L)
            left_index = (i - 1) % L
            right_index = (i + 1) % L
            if heights[left_index] < heights[i] or heights[right_index] < heights[i]:
                left_height = heights[left_index]
                right_height = heights[right_index]
                if left_height < right_height:
                    deposit_index = left_index
                elif right_height < left_height:
                    deposit_index = right_index
                else:
                    deposit_index = random.choice([left_index, right_index])
                heights[deposit_index] += 1
            else:
                heights[i] += 1
            t += 1
            if t % record_interval == 0:
                times.append(t)
                roughness_over_time.append(np.std(heights))
    return np.array(times), np.array(roughness_over_time), heights

L = 200
num_layers = 3
steps_per_layer = 500 * L
record_interval = 1000

times, roughness, _ = simulate_ballistic_deposition(L, num_layers, steps_per_layer, record_interval)
log_times = np.log10(times)
log_roughness = np.log10(roughness)
reg_rough_log = stats.linregress(log_times, log_roughness)
beta = reg_rough_log.slope
print("Growth exponent β (from roughness vs. time):", beta)

L_values = [50, 100, 200, 400]
saturation_roughness = []
saturation_time = []
for L_val in L_values:
    steps_per_layer_local = 500 * L_val
    times_val, roughness_val, _ = simulate_ballistic_deposition(L_val, num_layers, steps_per_layer_local, record_interval)
    saturation_roughness.append(roughness_val[-1])
    saturation_time.append(times_val[-1])
L_arr = np.array(L_values)
W_sat_arr = np.array(saturation_roughness)
t_sat_arr = np.array(saturation_time)
logL = np.log10(L_arr)
logW_sat = np.log10(W_sat_arr)
reg_alpha = stats.linregress(logL, logW_sat)
alpha = reg_alpha.slope
logt_sat = np.log10(t_sat_arr)
reg_z = stats.linregress(logL, logt_sat)
z = reg_z.slope
print("Roughness exponent α (from W_s ∼ L^α):", alpha)
print("Dynamic exponent z (from t_s ∼ L^z):", z)
print("Scaling relation check (β * z):", beta * z)

t_sat_L200 = num_layers * steps_per_layer
print("Number of deposited particles needed for saturation (L=200):", t_sat_L200)