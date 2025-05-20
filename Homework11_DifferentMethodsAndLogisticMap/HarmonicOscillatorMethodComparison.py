"""
Algorithms implemented
  1) Euler
  2) Euler–Cromer 
  3) Leapfrog
  4) Velocity-Verlet
  5) Beeman
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x: float) -> float:
    return -x

def energy(x: float, v: float) -> float:
    return 0.5 * (v**2 + x**2)

# 1) Euler (explicit / forward Euler)

def euler(x0, v0, h, tf):
    t = np.arange(0.0, tf, h)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)

    x[0], v[0] = x0, v0
    for n in range(len(t)-1):
        x[n+1] = x[n] + h * v[n]
        v[n+1] = v[n] + h * f(x[n])
    E[:] = energy(x, v)
    return t, x, v, E

# 2) Euler–Cromer  (also called Euler-Kramer)

def euler_kramer(x0, v0, h, tf):
    t = np.arange(0.0, tf, h)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)

    x[0], v[0] = x0, v0
    for n in range(len(t)-1):
        v[n+1] = v[n] + h * f(x[n])         
        x[n+1] = x[n] + h * v[n+1]          
    E[:] = energy(x, v)
    return t, x, v, E

# 3) Leapfrog  (frog-jump, staggered: v at half-steps)

def leapfrog(x0, v0, h, tf):
    t = np.arange(0.0, tf, h)
    x = np.zeros_like(t)
    v = np.zeros_like(t)     
    E = np.zeros_like(t)

    x[0] = x0
    v_half = v0 + 0.5*h*f(x0) 
    v[0] = v0

    for n in range(len(t)-1):
        x[n+1] = x[n] + h * v_half         
        v_half  = v_half + h * f(x[n+1])    
        v[n+1]  = v_half - 0.5*h*f(x[n+1])  
    E[:] = energy(x, v)
    return t, x, v, E

# 4) Velocity-Verlet  (second-order, symplectic)

def verlet(x0, v0, h, tf):
    t = np.arange(0.0, tf, h)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)

    x[0], v[0] = x0, v0
    a_n = f(x0)                         

    for n in range(len(t)-1):
        x[n+1] = x[n] + h*v[n] + 0.5*h**2 * a_n   
        a_next = f(x[n+1])                          
        v[n+1] = v[n] + 0.5*h*(a_n + a_next)       
        a_n    = a_next                            
    E[:] = energy(x, v)
    return t, x, v, E

# 5) Beeman

def beeman(x0, v0, h, tf):
    t = np.arange(0.0, tf, h)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)

    x[0], v[0] = x0, v0
    a_nm1      = f(x0)              
    v[1]       = v[0] + h * a_nm1
    x[1]       = x[0] + h * v[1]
    a_n        = f(x[1])               

    for n in range(1, len(t)-1):
        x_pred = x[n] + v[n]*h + (4*a_n - a_nm1)*h**2/6
        a_pred = f(x_pred)          
        
        v[n+1] = v[n] + h*(2*a_pred + 5*a_n - a_nm1)/6
        x[n+1] = x_pred

        a_nm1 = a_n
        a_n   = a_pred
    E[:] = energy(x, v)
    return t, x, v, E

def analytic(x0, v0, t):
    return x0*np.cos(t) + v0*np.sin(t)

if __name__ == "__main__":
    x0, v0 = 1.0, 0.0        
    tf     = 10.0           
    h_vals = [0.01, 0.05, 0.10, 0.50]  

    for h in h_vals:
        te, xe, ve, _ = euler(x0, v0, h, tf)
        tk, xk, vk, _ = euler_kramer(x0, v0, h, tf)
        tl, xl, vl, _ = leapfrog(x0, v0, h, tf)
        tv, xv, vv, _ = verlet(x0, v0, h, tf)
        tb, xb, vb, _ = beeman(x0, v0, h, tf)

        tref = np.arange(0.0, tf, h)
        xref = analytic(x0, v0, tref)

        plt.figure(figsize=(6,4))
        plt.plot(te,  xe,  'b.', ms=2, label='Euler')
        plt.plot(tk,  xk,  'g.', ms=2, label='Euler-Cromer')
        plt.plot(tl,  xl,  'y.', ms=2, label='Leapfrog')
        plt.plot(tv,  xv,  'c.', ms=2, label='Velocity-Verlet')
        plt.plot(tb,  xb,  'm.', ms=2, label='Beeman')
        plt.plot(tref, xref, 'k-', lw=1.4, label='Analytic')
        plt.title(f'Displacement vs time   (h = {h})')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(xe, ve, 'b.', ms=2, label='Euler')
        plt.plot(xk, vk, 'g.', ms=2, label='Euler-Cromer')
        plt.plot(xl, vl, 'y.', ms=2, label='Leapfrog')
        plt.plot(xv, vv, 'c.', ms=2, label='Velocity-Verlet')
        plt.plot(xb, vb, 'm.', ms=2, label='Beeman')
        plt.title(f'Phase Space Diagram   (h = {h})')
        plt.xlabel('x(t)')
        plt.ylabel('v(t)')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        _, _, _, Ee = euler(x0, v0, h, tf)
        _, _, _, Ek = euler_kramer(x0, v0, h, tf)
        _, _, _, El = leapfrog(x0, v0, h, tf)
        _, _, _, Ev = verlet(x0, v0, h, tf)
        _, _, _, Eb = beeman(x0, v0, h, tf)

        plt.figure(figsize=(6, 4))
        plt.plot(te, Ee, 'b-', lw=1, label='Euler')
        plt.plot(tk, Ek, 'g-', lw=1, label='Euler-Cromer')
        plt.plot(tl, El, 'y-', lw=1, label='Leapfrog')
        plt.plot(tv, Ev, 'c-', lw=1, label='Velocity-Verlet')
        plt.plot(tb, Eb, 'm-', lw=1, label='Beeman')
        plt.title(f'Energy vs Time   (h = {h})')
        plt.xlabel('t')
        plt.ylabel('E(t)')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()