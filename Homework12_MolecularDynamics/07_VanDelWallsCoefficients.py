import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

L = 30
N_particles = 100

def reject_outliers(data):
    stdev = np.std(data)
    mean = np.mean(data)
    maskMin = mean - stdev
    maskMax = mean + stdev
    return np.ma.masked_outside(data, maskMin, maskMax)

def init_particles(n, L, v_max):
    x_pos = np.linspace(0.5, L/2, int(n**0.5)+1)
    y_pos = np.linspace(0.5, L-0.5, int(n**0.5)+1)
    x_arr, y_arr = np.zeros(n), np.zeros(n)
    count = 0
    for y in y_pos:
        for x in x_pos:
            if count < n:
                x_arr[count] = x
                y_arr[count] = y
                count += 1
    vx_arr = np.random.uniform(-v_max, v_max, n)
    vy_arr = np.random.uniform(-v_max, v_max, n)
    return x_arr, y_arr, vx_arr, vy_arr

def kinetic_energy(vx, vy):
    return 0.5 * (vx**2 + vy**2)

def accel(x, y, L):
    n = len(x)
    fx = np.zeros((n, n))
    fy = np.zeros((n, n))
    half_L = L / 2
    for i in range(n):
        for j in range(i):
            dx = -(x[i] - x[j])
            dy = -(y[i] - y[j])
            if abs(dx) > half_L:
                dx -= L * np.sign(dx)
            if abs(dy) > half_L:
                dy -= L * np.sign(dy)
            r2 = dx**2 + dy**2
            r3 = r2**3
            r6 = r3**2
            force = -4 * (12/r6 - 6/r3) / r2
            fx[i][j] = force * dx
            fy[i][j] = force * dy
            fx[j][i] = -fx[i][j]
            fy[j][i] = -fy[i][j]
    return np.sum(fx, axis=1), np.sum(fy, axis=1), fx, fy

def virial(x, y, L):
    fx_matrix = accel(x, y, L)[2]
    fy_matrix = accel(x, y, L)[3]
    virial_sum = 0
    for i in range(len(x)):
        for j in range(i, len(x)):
            virial_sum += (x[i]-x[j])*fx_matrix[i][j] + (y[i]-y[j])*fy_matrix[i][j]
    return virial_sum / 2

def verlet_step(dt, steps, x, y, vx, vy, L):
    for _ in range(steps):
        ax, ay = accel(x, y, L)[0], accel(x, y, L)[1]
        x = (x + vx*dt + 0.5*ax*dt**2) % L
        y = (y + vy*dt + 0.5*ay*dt**2) % L
        ax_new, ay_new = accel(x, y, L)[0], accel(x, y, L)[1]
        vx += 0.5*dt*(ax + ax_new)
        vy += 0.5*dt*(ay + ay_new)
    return x, vx, y, vy

vi_range = np.arange(1.0, 2.0, 0.1)
T_values = np.zeros(len(vi_range))
P_values = np.zeros(len(vi_range))

for idx, v_max in enumerate(tqdm(vi_range)):
    x, y, vx, vy = init_particles(N_particles, L, v_max)
    vx -= np.mean(vx)
    vy -= np.mean(vy)
    T_arr = np.zeros(500)
    P_arr = np.zeros(500)
    for t in tqdm(range(500)):
        x, vx, y, vy = verlet_step(0.005, 10, x, y, vx, vy, L)
        T_arr[t] = np.sum(kinetic_energy(vx, vy)) / N_particles
        P_arr[t] = (N_particles * T_arr[t] + virial(x, y, L)) / (L**2)
    T_values[idx] = np.mean(reject_outliers(T_arr[-200:]))
    P_values[idx] = np.mean(reject_outliers(P_arr[-200:]))

def van_der_waals_model(T, a, b):
    V = L**2
    return (N_particles * T) / (V - N_particles * b) - a * N_particles**2 / V**2

params, _ = curve_fit(van_der_waals_model, T_values, P_values, p0=[1.0, 0.01])
a_fit, b_fit = params

print(f"\nFitted Van der Waals constants:")
print(f"a (reduced units) = {a_fit:.4f}")
print(f"b (reduced units) = {b_fit:.4f}")


plt.plot(T_values, P_values, 'bo', label='Simulation Data')
plt.plot(T_values, van_der_waals_model(T_values, *params), 'r-', label=f'Fit: P = (N T) / (V - N b) - a N^2 / V^2')
plt.xlabel('Temperature (T)')
plt.ylabel('Pressure (P)')
plt.title('Van der Waals Fit: Pressure vs. Temperature')
plt.legend()
plt.show()