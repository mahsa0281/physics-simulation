import numpy as np
import matplotlib.pyplot as plt

domain_size = 7

def setup_particles(num_atoms, box_size, max_velocity):
    grid_x = np.linspace(0.5, box_size / 2, int(np.sqrt(num_atoms)) + 1)
    grid_y = np.linspace(0.5, box_size - 0.5, int(np.sqrt(num_atoms)) + 1)
    pos_x = np.zeros(num_atoms)
    pos_y = np.zeros(num_atoms)
    count = 0
    for y in grid_y:
        for x in grid_x:
            if count < num_atoms:
                pos_x[count] = x
                pos_y[count] = y
                count += 1
    vel_x = np.random.uniform(-max_velocity, max_velocity, num_atoms)
    vel_y = np.random.uniform(-max_velocity, max_velocity, num_atoms)
    return pos_x, pos_y, vel_x, vel_y

def compute_forces(pos_x, pos_y, box_size):
    n = len(pos_x)
    fx_matrix = np.zeros((n, n))
    fy_matrix = np.zeros((n, n))
    half_box = box_size / 2
    for i in range(n):
        for j in range(i):
            dx = -(pos_x[i] - pos_x[j])
            dy = -(pos_y[i] - pos_y[j])
            if abs(dx) > half_box:
                dx -= box_size * np.sign(dx)
            if abs(dy) > half_box:
                dy -= box_size * np.sign(dy)
            r_squared = dx**2 + dy**2
            r_cubed = r_squared**3
            r_sixth = r_cubed**2
            force_factor = -4 * (12 / r_sixth - 6 / r_cubed) / r_squared
            fx_matrix[i, j] = force_factor * dx
            fy_matrix[i, j] = force_factor * dy
            fx_matrix[j, i] = -fx_matrix[i, j]
            fy_matrix[j, i] = -fy_matrix[i, j]
    return np.sum(fx_matrix, axis=1), np.sum(fy_matrix, axis=1)

def verlet_step(dt, steps, px, py, vx, vy, box_size):
    x = px
    y = py
    vx_temp = vx
    vy_temp = vy
    for _ in range(steps):
        ax, ay = compute_forces(px, py, box_size)[0], compute_forces(px, py, box_size)[1]
        x = (x + vx_temp * dt + 0.5 * ax * dt**2) % box_size
        y = (y + vy_temp * dt + 0.5 * ay * dt**2) % box_size
        ax_new, ay_new = compute_forces(px, py, box_size)[0], compute_forces(px, py, box_size)[1]
        vx_temp += 0.5 * dt * (ax + ax_new)
        vy_temp += 0.5 * dt * (ay + ay_new)
    return x, vx_temp, y, vy_temp

num_particles = 5
x_pos, y_pos, vx_init, vy_init = setup_particles(num_particles, domain_size, 0.7)

num_steps = 5000
x_data = np.zeros((num_steps, num_particles))
y_data = np.zeros((num_steps, num_particles))

for step in range(num_steps):
    x_pos, vx_init, y_pos, vy_init = verlet_step(0.001, 5, x_pos, y_pos, vx_init, vy_init, domain_size)
    x_data[step] = x_pos
    y_data[step] = y_pos

colors = ['b', 'g', 'r', 'y', 'k', 'c', 'm', 'brown', 'gray', 'orange']
for p in range(num_particles):
    plt.plot(x_data[:, p], y_data[:, p], '.', color=colors[p % len(colors)], markersize=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Particle Trajectories in 2D Box')
plt.show()

x_data = x_data.T
y_data = y_data.T

print("Final x-data shape:", x_data.shape)
print("Final y-data shape:", y_data.shape)
