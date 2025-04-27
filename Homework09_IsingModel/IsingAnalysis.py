import numpy as np, matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import ListedColormap

L_list = [32, 48, 64, 96, 128]         
T_list = np.linspace(3.5, 1.5, 61)      
n_eq   =  800                           
n_meas = 3000                        
np.random.seed(0)
spin_cmap = ListedColormap(['blue', 'yellow'])

def delta_E(lat,i,j,L):
    nb = ( lat[(i+1)%L, j] + lat[i-1, j] +
           lat[i, (j+1)%L] + lat[i, j-1] )
    return 2 * lat[i,j] * nb           

def sweep(lat,beta,boltz,L):
    for _ in range(L*L):
        i = np.random.randint(L); j = np.random.randint(L)
        dE = delta_E(lat,i,j,L)
        if dE<=0 or np.random.rand() < boltz[dE]:
            lat[i,j] = -lat[i,j]

def obs(lat):
    m = lat.mean()
    e = -np.sum(lat*( np.roll(lat,1,0)+np.roll(lat,-1,0)+
                      np.roll(lat,1,1)+np.roll(lat,-1,1) ))/(2.0*lat.size)
    return m, e                         

peaks = defaultdict(list)

for L in L_list:
    lat = np.random.choice([-1,1], size=(L,L))
    chi_max = Cv_max = m_below = 0.0

    for T in T_list:
        beta  = 1.0 / T
        boltz = {dE: np.exp(-beta*dE) for dE in (-8,-4,0,4,8)}

        for _ in range(n_eq):
            sweep(lat, beta, boltz, L)

        m_acc, e_acc = [], []
        for _ in range(n_meas):
            sweep(lat, beta, boltz, L)
            m, e = obs(lat)
            m_acc.append(m);  e_acc.append(e)

        m_arr = np.array(m_acc);  e_arr = np.array(e_acc)
        chi_T = L*L * np.var(m_arr) / T
        Cv_T  = L*L * np.var(e_arr) / T**2          

        if chi_T > chi_max: chi_max, Tchi = chi_T, T
        if Cv_T  > Cv_max : Cv_max , TCv  = Cv_T , T
        if T < 2.3: m_below = abs(m_arr.mean())     

    peaks['L']  .append(L)
    peaks['chi'].append(chi_max)
    peaks['Cv'] .append(Cv_max)
    peaks['Tchi'].append(Tchi)
    peaks['m']  .append(m_below)
    print(f"L={L:3d}:  χ_max={chi_max:7.2f}   Cv_max={Cv_max:5.3f}")

Ls      = np.array(peaks['L'])
chi_max = np.array(peaks['chi'])
Cv_max  = np.array(peaks['Cv'])
m_below = np.array(peaks['m'])

γ_over_ν  = np.polyfit(np.log(Ls), np.log(chi_max), 1)[1]
β_over_ν  =-np.polyfit(np.log(Ls), np.log(m_below), 1)[1]
c0        = np.polyfit(np.log(Ls), Cv_max, 1)[0]

Tc_exact  = 2.269185
inv_ν     = np.polyfit(np.log(Ls),
                       np.log(np.abs(np.array(peaks['Tchi'])-Tc_exact)), 1)[1]

print("\n--- rough finite-size exponents (5 sizes) ---")
print(f"γ/ν  ≈ {γ_over_ν:5.3f}  (exact 1.750)")
print(f"β/ν  ≈ {β_over_ν:5.3f}  (exact 0.125)")
print(f"ν     ≈ {1/inv_ν:5.3f}  (exact 1.000)")
print(f"c₀    ≈ {c0:5.3f}  (Cv_max ≃ c₀ ln L)")

plt.figure(figsize=(5,4))
for L, Tchi, cmax in zip(Ls, peaks['Tchi'], Cv_max):
    plt.scatter(Tchi, L, label=f"L={L}")
plt.xlabel("T (χ peak)"); plt.ylabel("L"); plt.title("Peak drift")
plt.legend(); plt.tight_layout(); plt.show()