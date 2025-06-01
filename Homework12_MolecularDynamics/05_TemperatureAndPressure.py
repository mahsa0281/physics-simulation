import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 30

def reject_outliers(data):
    stdev = np.std(data)
    mean = np.mean(data)
    maskMin = mean - 2 * stdev
    maskMax = mean + 2 * stdev
    mask = np.ma.masked_outside(data, maskMin, maskMax)
    print(f"Masked values outside {maskMin:.3f} and {maskMax:.3f}")
    return mask

def init_particles(n, L, v_max):
    x_pos = np.linspace(0.5, L/2, int(n**0.5)+1)
    y_pos = np.linspace(0.5, L-0.5, int(n**0.5)+1)
    pos_x, pos_y = np.zeros(n), np.zeros(n)
    count = 0
    for y in y_pos:
        for x in x_pos:
            if count < n:
                pos_x[count] = x
                pos_y[count] = y
                count += 1
    vel_x = np.random.uniform(-v_max, v_max, n)
    vel_y = np.random.uniform(-v_max, v_max, n)
    return pos_x, pos_y, vel_x, vel_y

def kinetic_energy(vel_x, vel_y):
    return 0.5 * (vel_x**2 + vel_y**2)

def accel(pos_x, pos_y, L):
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
            force = -4 * (12 / r_sixth - 6 / r_cubed) / r_sq
            fx[i][j] = force * dx
            fy[i][j] = force * dy
            fx[j][i] = -fx[i][j]
            fy[j][i] = -fy[i][j]
    return np.sum(fx, axis=1), np.sum(fy, axis=1), fx, fy

def virial_term(pos_x, pos_y, L):
    fx_mat = accel(pos_x, pos_y, L)[2]
    fy_mat = accel(pos_x, pos_y, L)[3]
    total = 0
    for i in range(len(pos_x)):
        for j in range(i, len(pos_x)):
            total += (pos_x[i]-pos_x[j])*fx_mat[i][j] + (pos_y[i]-pos_y[j])*fy_mat[i][j]
    return total / 2

def verlet(h, N, pos_x, pos_y, vel_x, vel_y, L):
    x, y = pos_x, pos_y
    vx, vy = vel_x, vel_y
    for _ in range(N):
        ax, ay = accel(x, y, L)[0], accel(x, y, L)[1]
        x = (x + vx*h + 0.5*ax*h**2) % L
        y = (y + vy*h + 0.5*ay*h**2) % L
        ax_new, ay_new = accel(x, y, L)[0], accel(x, y, L)[1]
        vx += 0.5*h*(ax + ax_new)
        vy += 0.5*h*(ay + ay_new)
    return x, vx, y, vy

N_particles = 100
x, y, vx, vy = init_particles(N_particles, L, 1.0)
vx -= np.mean(vx)
vy -= np.mean(vy)

timesteps = 500
T_arr = np.zeros(timesteps)
P_arr = np.zeros(timesteps)

for step in tqdm(range(timesteps)):
    x, vx, y, vy = verlet(0.005, 10, x, y, vx, vy, L)
    T_arr[step] = np.sum(kinetic_energy(vx, vy)) / N_particles
    P_arr[step] = (N_particles * T_arr[step] + virial_term(x, y, L)) / (L*L)

plt.plot(T_arr, 'r-')
plt.xlabel('Time step')
plt.ylabel('Temperature')
plt.title('Temperature over Time')
plt.show()

Temp_avg = np.mean(reject_outliers(T_arr[-200:]))
print(f"Average Temperature: {Temp_avg:.3f}")

plt.plot(P_arr, 'b-')
plt.xlabel('Time step')
plt.ylabel('Pressure')
plt.title('Pressure over Time')
plt.show()

Pressure_avg = np.mean(reject_outliers(P_arr[-200:]))
print(f"Average Pressure: {Pressure_avg:.3f}")