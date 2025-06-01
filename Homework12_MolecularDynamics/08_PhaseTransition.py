import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 30
N_particles = 100
scale_factor = 0.8
repeats = 10
timesteps = 500

def initialize_particles(n, L, v_max):
    x_positions = np.zeros(n)
    y_positions = np.zeros(n)
    x_grid = np.linspace(0.5, L/2, int(n**0.5)+1)
    y_grid = np.linspace(0.5, L-0.5, int(n**0.5)+1)
    idx = 0
    for y in y_grid:
        for x in x_grid:
            if idx < n:
                x_positions[idx] = x
                y_positions[idx] = y
                idx += 1
    vx = np.random.uniform(-v_max, v_max, n)
    vy = np.random.uniform(-v_max, v_max, n)
    return x_positions, y_positions, vx, vy

def compute_forces(x, y, L):
    n = len(x)
    fx = np.zeros((n,n))
    fy = np.zeros((n,n))
    rc = L/2
    for i in range(n):
        for j in range(i):
            dx = -(x[i]-x[j])
            dy = -(y[i]-y[j])
            if abs(dx) > rc:
                dx -= L * np.sign(dx)
            if abs(dy) > rc:
                dy -= L * np.sign(dy)
            r2 = dx**2 + dy**2
            r6 = r2**3
            r12 = r6**2
            f_scalar = -4 * (12/r12 - 6/r6) / r2
            fx[i][j] = f_scalar * dx
            fy[i][j] = f_scalar * dy
            fx[j][i] = -fx[i][j]
            fy[j][i] = -fy[i][j]
    return np.sum(fx, axis=1), np.sum(fy, axis=1)

def verlet_step(dt, N_steps, x, y, vx, vy, L):
    for _ in range(N_steps):
        ax, ay = compute_forces(x, y, L)
        x = (x + vx*dt + 0.5*ax*dt**2) % L
        y = (y + vy*dt + 0.5*ay*dt**2) % L
        ax_new, ay_new = compute_forces(x, y, L)
        vx += 0.5*dt*(ax + ax_new)
        vy += 0.5*dt*(ay + ay_new)
    return x, y, vx, vy

def kinetic_energy(vx, vy):
    return 0.5 * np.sum(vx**2 + vy**2)

def potential_energy(x, y, L):
    n = len(x)
    pot = 0
    rc = L/2
    for i in range(n):
        for j in range(i):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            if abs(dx) > rc:
                dx -= L * np.sign(dx)
            if abs(dy) > rc:
                dy -= L * np.sign(dy)
            r2 = dx**2 + dy**2
            r6 = r2**3
            r12 = r6**2
            pot += 4*(1/r12 - 1/r6)
    return pot

x, y, vx, vy = initialize_particles(N_particles, L, 2.0)
vx -= np.mean(vx)
vy -= np.mean(vy)

energy_list = []
temperature_list = []

for repeat in tqdm(range(repeats)):
    energies = []
    temperatures = []
    for t in tqdm(range(timesteps)):
        x, y, vx, vy = verlet_step(0.005, 10, x, y, vx, vy, L)
        ke = kinetic_energy(vx, vy)
        pe = potential_energy(x, y, L)
        energies.append(ke + pe)
        temperatures.append(ke / N_particles)
    energy_list.append(np.mean(energies[-200:]))
    temperature_list.append(np.mean(temperatures[-200:]))
    
    vx *= scale_factor
    vy *= scale_factor

plt.plot(temperature_list, energy_list, 'bo')
plt.title('Phase Transition: Energy vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.show()