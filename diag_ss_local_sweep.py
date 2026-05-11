"""diag_ss_local_sweep.py — sweep по (n, κ, R) для подбора параметров
fig_ss_local_quadratic. Цель: найти такие (n, κ, R), при которых
PSB vs SS-PSB на чистой квадратике без Armijo даёт визуально читаемый
контраст пре-асимптотической фазы.

Для каждой комбинации печатает медианное число итераций до tol=1e-12
по 30 стартам (paired-design).
"""
from __future__ import annotations

import warnings
import numpy as np
from numpy.linalg import norm
from diag_ss_local_quadratic import quadratic, run_traced

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    rng = np.random.default_rng(20260503)
    n_dirs = 30
    tol = 1e-12

    configs = []
    for n in [10, 20, 50, 100]:
        for kappa in [10.0, 100.0, 1000.0, 10000.0]:
            for R in [0.1, 0.3, 1.0, 3.0]:
                configs.append((n, kappa, R))

    methods = ['psb', 'ss_psb']

    U_by_n = {}
    for n in {c[0] for c in configs}:
        Z = rng.standard_normal((n_dirs, n))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        U_by_n[n] = Z

    print(f"{'n':>4} {'κ':>7} {'R':>5}  {'PSB med':>8} {'SS-PSB med':>11} {'ratio':>6}  {'PSB conv':>9} {'SS conv':>9}")
    print("-"*70)

    for n, kappa, R in configs:
        max_iter = max(20*n, 200)
        prob = quadratic(n=n, kappa=kappa)
        results = {}
        for mk in methods:
            iters_to_conv = []
            n_conv = 0
            for di in range(n_dirs):
                x0 = R * U_by_n[n][di]
                h = run_traced(prob, x0, mk, max_iter=max_iter, tol=tol)
                if h[-1] <= tol:
                    iters_to_conv.append(len(h)-1)
                    n_conv += 1
            results[mk] = (iters_to_conv, n_conv)

        psb_med = int(np.median(results['psb'][0])) if results['psb'][0] else -1
        ss_med  = int(np.median(results['ss_psb'][0])) if results['ss_psb'][0] else -1
        ratio = (psb_med / ss_med) if (psb_med > 0 and ss_med > 0) else 0.0
        psb_c, ss_c = results['psb'][1], results['ss_psb'][1]
        print(f"{n:>4} {kappa:>7g} {R:>5g}  {psb_med:>8} {ss_med:>11} {ratio:>6.2f}  "
              f"{psb_c:>4}/{n_dirs:<4} {ss_c:>4}/{n_dirs:<4}")


if __name__ == "__main__":
    main()
