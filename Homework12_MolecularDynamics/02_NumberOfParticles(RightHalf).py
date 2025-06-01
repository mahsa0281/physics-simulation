import numpy as np
import matplotlib.pyplot as plt

box_length = 30

def initialize_system(n_particles, box_size, max_velocity):
    grid_x = np.linspace(0.5, box_size / 2, int(np.sqrt(n_particles)) + 1)
    grid_y = np.linspace(0.5, box_size - 0.5, int(np.sqrt(n_particles)) + 1)
    x_positions = np.zeros(n_particles)
    y_positions = np.zeros(n_particles)
    idx = 0
    for y in grid_y:
        for x in grid_x:
            if idx < n_particles:
                x_positions[idx] = x
                y_positions[idx] = y
                idx += 1
    vx_values = np.random.uniform(-max_velocity, max_velocity, n_particles)
    vy_values = np.random.uniform(-max_velocity, max_velocity, n_particles)
    return x_positions, y_positions, vx_values, vy_values

def calculate_forces(x_positions, y_positions, box_size):
    n = len(x_positions)
    fx_matrix = np.zeros((n, n))
    fy_matrix = np.zeros((n, n))
    cutoff = box_size / 2
    for i in range(n):
        for j in range(i):
            dx = -(x_positions[i] - x_positions[j])
            dy = -(y_positions[i] - y_positions[j])
            if abs(dx) > cutoff:
                dx -= box_size * np.sign(dx)
            if abs(dy) > cutoff:
                dy -= box_size * np.sign(dy)
            r2 = dx**2 + dy**2
            r3 = r2**3
            r6 = r3**2
            force_value = -4 * (12 / r6 - 6 / r3) / r2
            fx_matrix[i, j] = force_value * dx
            fy_matrix[i, j] = force_value * dy
            fx_matrix[j, i] = -fx_matrix[i, j]
            fy_matrix[j, i] = -fy_matrix[i, j]
    return np.sum(fx_matrix, axis=1), np.sum(fy_matrix, axis=1)

def verlet_update(dt, steps, pos_x, pos_y, vel_x, vel_y, box_size):
    x = pos_x
    y = pos_y
    vx = vel_x
    vy = vel_y
    for _ in range(steps):
        ax, ay = calculate_forces(x, y, box_size)
        x = (x + vx * dt + 0.5 * ax * dt**2) % box_size
        y = (y + vy * dt + 0.5 * ay * dt**2) % box_size
        ax_new, ay_new = calculate_forces(x, y, box_size)
        vx += 0.5 * dt * (ax + ax_new)
        vy += 0.5 * dt * (ay + ay_new)
    return x, vx, y, vy

num_atoms = 100
x_coords, y_coords, vx_coords, vy_coords = initialize_system(num_atoms, box_length, 2.0)

num_frames = 500
x_history = np.zeros((num_frames, num_atoms))
y_history = np.zeros((num_frames, num_atoms))
right_half_ratio = np.zeros(num_frames)

for frame in range(num_frames):
    x_coords, vx_coords, y_coords, vy_coords = verlet_update(0.005, 10, x_coords, y_coords, vx_coords, vy_coords, box_length)
    x_history[frame] = x_coords
    y_history[frame] = y_coords
    count_in_left = np.sum(x_coords < box_length / 2)
    right_half_ratio[frame] = count_in_left / num_atoms

for t in range(num_frames):
    plt.plot(t, right_half_ratio[t], 'bo')
plt.xlabel('Time Step')
plt.ylabel('Fraction of Particles in Left Half')
plt.title('Particles Distribution Over Time')
plt.show()