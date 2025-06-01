import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

box_length = 30

def compute_potential(pos_x, pos_y, box_size):
    n = len(pos_x)
    potential_matrix = np.zeros((n, n))
    half_box = box_size / 2
    for i in range(n):
        for j in range(i):
            dx = -(pos_x[i] - pos_x[j])
            dy = -(pos_y[i] - pos_y[j])
            if abs(dx) > half_box:
                dx -= box_size * np.sign(dx)
            if abs(dy) > half_box:
                dy -= box_size * np.sign(dy)
            r_sq = dx**2 + dy**2
            r_cubed = r_sq**3
            r_sixth = r_cubed**2
            potential_val = 4 * (1 / r_sixth - 1 / r_cubed)
            potential_matrix[i, j] = potential_val
            potential_matrix[j, i] = potential_val
    return np.sum(potential_matrix, axis=1)

def compute_kinetic(vel_x, vel_y):
    return 0.5 * (vel_x**2 + vel_y**2)

def initialize_particles(n_particles, box_size, max_velocity):
    grid_x = np.linspace(0.5, box_size / 2, int(np.sqrt(n_particles)) + 1)
    grid_y = np.linspace(0.5, box_size - 0.5, int(np.sqrt(n_particles)) + 1)
    pos_x = np.zeros(n_particles)
    pos_y = np.zeros(n_particles)
    index = 0
    for y in grid_y:
        for x in grid_x:
            if index < n_particles:
                pos_x[index] = x
                pos_y[index] = y
                index += 1
    vel_x = np.random.uniform(-max_velocity, max_velocity, n_particles)
    vel_y = np.random.uniform(-max_velocity, max_velocity, n_particles)
    return pos_x, pos_y, vel_x, vel_y

def compute_forces(pos_x, pos_y, box_size):
    n = len(pos_x)
    fx = np.zeros((n, n))
    fy = np.zeros((n, n))
    half_box = box_size / 2
    for i in range(n):
        for j in range(i):
            dx = -(pos_x[i] - pos_x[j])
            dy = -(pos_y[i] - pos_y[j])
            if abs(dx) > half_box:
                dx -= box_size * np.sign(dx)
            if abs(dy) > half_box:
                dy -= box_size * np.sign(dy)
            r_sq = dx**2 + dy**2
            r_cubed = r_sq**3
            r_sixth = r_cubed**2
            force_value = -4 * (12 / r_sixth - 6 / r_cubed) / r_sq
            fx[i, j] = force_value * dx
            fy[i, j] = force_value * dy
            fx[j, i] = -fx[i, j]
            fy[j, i] = -fy[i, j]
    return np.sum(fx, axis=1), np.sum(fy, axis=1)

def verlet_step(dt, steps, pos_x, pos_y, vel_x, vel_y, box_size):
    x = pos_x
    y = pos_y
    vx = vel_x
    vy = vel_y
    for _ in range(steps):
        ax, ay = compute_forces(x, y, box_size)
        x = (x + vx * dt + 0.5 * ax * dt**2) % box_size
        y = (y + vy * dt + 0.5 * ay * dt**2) % box_size
        ax_new, ay_new = compute_forces(x, y, box_size)
        vx += 0.5 * dt * (ax + ax_new)
        vy += 0.5 * dt * (ay + ay_new)
    return x, vx, y, vy

num_particles = 100
pos_x, pos_y, vel_x, vel_y = initialize_particles(num_particles, box_length, 2.0)
vel_x -= np.mean(vel_x)
vel_y -= np.mean(vel_y)

timesteps = 500
potential_energy = np.zeros(timesteps)
kinetic_energy = np.zeros(timesteps)

for t in tqdm(range(timesteps)):
    pos_x, vel_x, pos_y, vel_y = verlet_step(0.005, 10, pos_x, pos_y, vel_x, vel_y, box_length)
    potential_energy[t] = np.sum(compute_potential(pos_x, pos_y, box_length)) / 2
    kinetic_energy[t] = np.sum(compute_kinetic(vel_x, vel_y))

for t in range(timesteps):
    plt.plot(t, potential_energy[t], 'bo', label='Potential Energy' if t == 0 else "")
    plt.plot(t, kinetic_energy[t], 'ro', label='Kinetic Energy' if t == 0 else "")
    plt.plot(t, potential_energy[t] + kinetic_energy[t], 'go', label='Total Energy' if t == 0 else "")

plt.xlabel('Time Step')
plt.ylabel('Energy')
plt.title('Potential, Kinetic, and Total Energy over Time')
plt.legend()
plt.show()