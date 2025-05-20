import numpy as np

def logistic_tail(r: float,
                  n_iter: int = 20000,
                  keep: int = 200,
                  x0: float = 0.2) -> np.ndarray:
    x = x0
    buf = np.empty(keep)
    for i in range(n_iter):
        x = 4 * r * x * (1.0 - x)
        if i >= n_iter - keep:
            buf[i - (n_iter - keep)] = x
    return np.round(buf, 6)         

r_grid   = np.arange(0.70, 0.95, 1e-4)  
targets  = [1, 2, 4, 8, 16, 32]        
min_sep  = 5e-4                           

bif_r    = []                           
idx_tgt  = 0                           
prev_r   = -1.0

print("Bifurcation points:")
for r in r_grid:
    tail = logistic_tail(r)
    period = len(np.unique(tail))

    if idx_tgt >= len(targets):       
        break

    target = targets[idx_tgt]
    if period == target and abs(r - prev_r) > min_sep:
        bif_r.append(r)
        prev_r = r
        idx_tgt += 1
        print(f"  period {period:>2}  at  r = {r:.10f}")

print("\nFeigenbaum-δ estimates:")
for k in range(1, len(bif_r) - 1):
    delta = (bif_r[k] - bif_r[k-1]) / (bif_r[k+1] - bif_r[k])
    print(f"  δ_{k} = {delta:.6f}")