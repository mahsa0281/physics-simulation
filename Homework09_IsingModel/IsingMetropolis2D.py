import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  

L        = 80                
T_list   = [5.0, 2.0, 1.2]   
n_eq     = 50                 
n_meas   = 100              
np.random.seed(1)             

def delta_E(lat, i, j):
    s   = lat[i, j]
    nbr = ( lat[(i+1)%L, j] + lat[i-1, j] +
            lat[i, (j+1)%L] + lat[i, j-1] )
    return 2 * s * nbr    

def sweep(lat, beta, boltz):
    for _ in range(L*L):
        i, j  = np.random.randint(L), np.random.randint(L)
        dE    = delta_E(lat, i, j)
        if dE <= 0 or np.random.rand() < boltz[dE]:
            lat[i, j] = -lat[i, j]

def energy_per_spin(lat):
    nbr_sum = ( np.roll(lat,  1, 0) + np.roll(lat, -1, 0) +
                np.roll(lat,  1, 1) + np.roll(lat, -1, 1) )
    return -np.sum(lat * nbr_sum) / (2 * L * L)

lattice = np.random.choice([-1, 1], size=(L, L))

temps, m_av, chi, e_av, cv = [], [], [], [], []

for T in T_list:
    beta  = 1.0 / T
    boltz = {dE: np.exp(-beta*dE) for dE in (-8, -4, 0, 4, 8)}

    for _ in range(n_eq):
        sweep(lattice, beta, boltz)

    mags, enes = [], []
    for _ in range(n_meas):
        sweep(lattice, beta, boltz)
        mags.append(abs(lattice.mean()))
        enes.append(energy_per_spin(lattice))

    temps.append(T)
    m_av.append(np.mean(mags))
    chi.append(L*L * np.var(mags) / T)
    e_av.append(np.mean(enes))
    cv.append(np.var(enes) / (T*T))

    spin_cmap = ListedColormap(['blue', 'yellow'])    
    plt.figure(figsize=(4,4))
    plt.imshow(lattice, cmap=spin_cmap, vmin=-1, vmax=1)   
    plt.title(f"Spin configuration  (T = {T})")
    plt.axis('off')
    plt.show()

invT = 1/np.array(temps)

plt.figure(figsize=(6,4))
plt.plot(invT, m_av, 'o-')
plt.xlabel(r'$T^{-1}$');  plt.ylabel(r'$\langle |m| \rangle$')
plt.title('Magnetisation vs 1/T');  plt.show()

plt.figure(figsize=(6,4))
plt.plot(invT, chi, 'o-g')
plt.xlabel(r'$T^{-1}$');  plt.ylabel(r'$\chi$')
plt.title('Susceptibility vs 1/T');  plt.show()

plt.figure(figsize=(6,4))
plt.plot(invT, e_av, 'o-r')
plt.xlabel(r'$T^{-1}$');  plt.ylabel(r'$\langle E \rangle$ per spin')
plt.title('Energy vs 1/T');  plt.show()

plt.figure(figsize=(6,4))
plt.plot(invT, cv, 'o-b')
plt.xlabel(r'$T^{-1}$');  plt.ylabel(r'$C_v$')
plt.title('Heat capacity vs 1/T');  plt.show()