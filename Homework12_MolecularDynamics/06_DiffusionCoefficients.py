import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 30
N_particles = 100

def init(n, L, v_max):
    x_pos = np.linspace(0.5, L/2, int(n**0.5)+1)
    y_pos = np.linspace(0.5, L-0.5, int(n**0.5)+1)
    x_arr = np.zeros(n)
    y_arr = np.zeros(n)
    num = 0
    for j in range(len(y_pos)):
        for i in range(len(x_pos)):
            if num < n:
                x_arr[num] = x_pos[i]
                y_arr[num] = y_pos[j]
                num += 1
    vx_arr = np.random.uniform(-v_max, v_max, n)
    vy_arr = np.random.uniform(-v_max, v_max, n)
    return x_arr, y_arr, vx_arr, vy_arr

def accel(x_arr, y_arr, L):
    n = len(x_arr)
    fx_arr = np.zeros((n,n))
    fy_arr = np.zeros((n,n))
    rc = L/2
    for i in range(n):
        for j in range(i):
            deltax = -(x_arr[i]-x_arr[j])
            deltay = -(y_arr[i]-y_arr[j])
            if abs(deltax) > rc:
                deltax -= L * np.sign(deltax)
            if abs(deltay) > rc:
                deltay -= L * np.sign(deltay)
            r2 = deltax**2 + deltay**2
            r3 = r2**3
            r6 = r3**2
            tmp = -4 * (12/r6 - 6/r3) / r2
            fx_arr[i][j] = tmp * deltax
            fy_arr[i][j] = tmp * deltay
            fx_arr[j][i] = -fx_arr[i][j]
            fy_arr[j][i] = -fy_arr[i][j]
    fx_tot = np.sum(fx_arr, axis=1)
    fy_tot = np.sum(fy_arr, axis=1)
    return fx_tot, fy_tot

def verle(h, N, x_0, y_0, vx_0, vy_0, L):
    rx = x_0
    vx = vx_0
    ry = y_0
    vy = vy_0
    for n in range(N):
        ax, ay = accel(rx, ry, L)
        rx = (rx + vx*h + 0.5*ax*h**2) % L
        ry = (ry + vy*h + 0.5*ay*h**2) % L
        ax_new, ay_new = accel(rx, ry, L)
        vx += 0.5*h*(ax + ax_new)
        vy += 0.5*h*(ay + ay_new)
    return rx, vx, ry, vy

def compute_MSD(traj, L):
    N_particles = traj.shape[1]
    T_steps = traj.shape[0]
    MSD = np.zeros(T_steps)
    for dt in range(T_steps):
        displacements = []
        for i in range(T_steps - dt):
            dx = traj[i+dt] - traj[i]
            dx = (dx + L/2) % L - L/2  
            displacements.append(np.mean(dx**2))
        MSD[dt] = np.mean(displacements)
    return MSD

timesteps = 1000
x_arr, y_arr, vx_arr, vy_arr = init(N_particles, L, 1.0)
vx_arr -= np.mean(vx_arr)
vy_arr -= np.mean(vy_arr)

x_traj = np.zeros((timesteps, N_particles))
y_traj = np.zeros((timesteps, N_particles))

for t in tqdm(range(timesteps)):
    x_arr, vx_arr, y_arr, vy_arr = verle(0.005, 10, x_arr, y_arr, vx_arr, vy_arr, L)
    x_traj[t] = x_arr
    y_traj[t] = y_arr

MSD_x = compute_MSD(x_traj, L)
MSD_y = compute_MSD(y_traj, L)
MSD_total = MSD_x + MSD_y

t_values = np.arange(len(MSD_total)) * 0.005 * 10
slope, intercept = np.polyfit(t_values, MSD_total, 1)
D_estimate = slope / 4

print(f"Estimated Diffusion Coefficient D (reduced units): {D_estimate:.4f}")

plt.plot(t_values, MSD_total, label='MSD')
plt.plot(t_values, slope*t_values + intercept, 'r--', label=f'Fit: D = {D_estimate:.4f}')
plt.xlabel('Time')
plt.ylabel('MSD')
plt.title('Mean Squared Displacement and Diffusion Coefficient')
plt.legend()
plt.show()