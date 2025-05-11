import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Q0 = 1.0        
R  = 1.0         
C  = 1.0         
T_final = 5.0     
h = 0.1           

def dQ_dt(Q):
    return -Q / (R * C)

def Q_exact(t):
    return Q0 * np.exp(-t / (R * C))

t = np.arange(0.0, T_final + h, h)    
N = len(t)

Q_eul = np.empty(N)
Q_eul[0] = Q0

for n in range(1, N):
    Q_eul[n] = Q_eul[n-1] + h * dQ_dt(Q_eul[n-1])      
plt.figure(figsize=(6,4))
plt.plot(t, Q_exact(t),  label="Exact solution")
plt.plot(t, Q_eul, 'o--',label="Euler approximation")
plt.xlabel("Time  t  [s]")
plt.ylabel("Charge  Q(t)  [C]")
plt.title("Discharge of a capacitor in an RC circuit")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

df = pd.DataFrame({
    't (s)': t,
    'Q_exact': Q_exact(t),
    'Q_Euler': Q_eul,
    'abs error': np.abs(Q_exact(t) - Q_eul)
})
print(df.head(10))

print("\nMax absolute error on [0, {:.1f}] s  = {:.4e} C".format(T_final, df['abs error'].max()))