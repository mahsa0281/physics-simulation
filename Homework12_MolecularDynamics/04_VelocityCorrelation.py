import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  

box_size = 30

def potential_energy(pos_x, pos_y, L):
    n = len(pos_x)
    energy_matrix = np.zeros((n, n))
    half_L = L / 2
    for i in range(n):
        for j in range(i):
            dx = (pos_x[i] - pos_x[j])
            dy = (pos_y[i] - pos_y[j])
            if abs(dx) > half_L:
                dx -= L * np.sign(dx)
            if abs(dy) > half_L:
                dy -= L * np.sign(dy)
            r_sq = dx**2 + dy**2
            r_cubed = r_sq**3
            r_sixth = r_cubed**2
            value = 4 * (1 / r_sixth - 1 / r_cubed)
            energy_matrix[i, j] = value
            energy_matrix[j, i] = value
    return np.sum(energy_matrix, axis=1)

def kinetic_energy(vel_x, vel_y):
    return 0.5 * (vel_x**2 + vel_y**2)

def initialize(n_particles, L, v_max):
    grid_x = np.linspace(0.5, L / 2, int(np.sqrt(n_particles)) + 1)
    grid_y = np.linspace(0.5, L - 0.5, int(np.sqrt(n_particles)) + 1)
    pos_x = np.zeros(n_particles)
    pos_y = np.zeros(n_particles)
    count = 0
    for y in grid_y:
        for x in grid_x:
            if count < n_particles:
                pos_x[count] = x
                pos_y[count] = y
                count += 1
    vel_x = np.random.uniform(-v_max, v_max, n_particles)
    vel_y = np.random.uniform(-v_max, v_max, n_particles)
    return pos_x, pos_y, vel_x, vel_y

def compute_forces(pos_x, pos_y, L):
    n = len(pos_x)
    fx = np.zeros((n, n))
    fy = np.zeros((n, n))
    half_L = L / 2
    for i in range(n):
        for j in range(i):
            dx = -(pos_x[i] - pos_x[j])
            dy = -(pos_y[i] - pos_y[j])
            if abs(dx) > half_L:
                dx -= L * np.sign(dx)
            if abs(dy) > half_L:
                dy -= L * np.sign(dy)
            r_sq = dx**2 + dy**2
            r_cubed = r_sq**3
            r_sixth = r_cubed**2
            f_value = -4 * (12 / r_sixth - 6 / r_cubed) / r_sq
            fx[i, j] = f_value * dx
            fy[i, j] = f_value * dy
            fx[j, i] = -fx[i, j]
            fy[j, i] = -fy[i, j]
    return np.sum(fx, axis=1), np.sum(fy, axis=1)

def verlet_update(dt, n_steps, pos_x, pos_y, vel_x, vel_y, L):
    x = pos_x
    y = pos_y
    vx = vel_x
    vy = vel_y
    for _ in range(n_steps):
        ax, ay = compute_forces(x, y, L)
        x = (x + vx * dt + 0.5 * ax * dt**2) % L
        y = (y + vy * dt + 0.5 * ay * dt**2) % L
        ax_new, ay_new = compute_forces(x, y, L)
        vx += 0.5 * dt * (ax + ax_new)
        vy += 0.5 * dt * (ay + ay_new)
    return x, vx, y, vy

def autocorr(j, data_array):
    sum_prod = 0
    for k in range(len(data_array) - j):
        sum_prod += data_array[k] * data_array[(k + j) % len(data_array)]
    mean_prod = sum_prod / (len(data_array) - j)
    mean_square = np.mean(data_array)**2
    variance = np.std(data_array)**2
    return (mean_prod - mean_square) / variance

N = 100
pos_x, pos_y, vel_x, vel_y = initialize(N, box_size, 2.0)
vel_x -= np.mean(vel_x)
vel_y -= np.mean(vel_y)

num_steps = 500
vel_traj_x = np.zeros((num_steps, N))
vel_traj_y = np.zeros((num_steps, N))

for t in tqdm(range(num_steps)):
    pos_x, vel_x, pos_y, vel_y = verlet_update(0.005, 10, pos_x, pos_y, vel_x, vel_y, box_size)
    vel_traj_x[t] = vel_x
    vel_traj_y[t] = vel_y

vel_traj_x = np.transpose(vel_traj_x)
vel_traj_y = np.transpose(vel_traj_y)

corr_x = np.array([[autocorr(t, vel_traj_x[i]) for i in range(N)] for t in range(num_steps)])
corr_avg_x = np.array([np.mean(corr_x[t]) for t in range(num_steps)])
corr_y = np.array([[autocorr(t, vel_traj_y[i]) for i in range(N)] for t in range(num_steps)])
corr_avg_y = np.array([np.mean(corr_y[t]) for t in range(num_steps)])

plt.plot(corr_avg_x, 'r-', label='X-Velocity Correlation')
plt.xlabel('Time step')
plt.ylabel('Correlation')
plt.title('Autocorrelation of X-Velocity')
plt.legend()
plt.show()

plt.plot(corr_avg_y, 'b-', label='Y-Velocity Correlation')
plt.xlabel('Time step')
plt.ylabel('Correlation')
plt.title('Autocorrelation of Y-Velocity')
plt.legend()
plt.show()