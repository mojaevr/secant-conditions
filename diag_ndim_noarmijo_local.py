"""diag_ndim_noarmijo_local.py — локальная сходимость БЕЗ Armijo.

Дополняет diag_ndim_noarmijo.py: вместо глобальных стартов на сфере
радиуса R_0=||x_0|| (где локальная теорема не работает) проверяет
сходимость в *окрестности* x^*:

  * существующие 5 задач (Розенброк chained, ext.Розенброк, ECV n=10/20/50)
    с малыми радиусами R ∈ {0.3, 0.1, 0.03} вокруг x^*
    (для Розенброков x^* = 1, для ECV x^* = 0);

  * контролируемые квадратичные задачи: f(x) = (1/2) x^T A x,
    A = Q diag(λ_i) Q^T, λ_i ∈ [1, κ], κ ∈ {10, 100}, n=10, x^* = 0;
    учебниковый локальный случай — SR1 сходится в ≤ n+1 шагов
    в точной арифметике.

Всё остальное — как в diag_ndim_noarmijo.py: полный QN-шаг
x_{k+1}=x_k+d_k, paired-design (тот же seed 20260503), 50 направлений
на S^{n-1}, p=5, max_iter=500, tol=1e-8.
"""
from __future__ import annotations

import os
import time
import numpy as np
from numpy.linalg import norm

from diag_ndim_stat import (
    extended_rosenbrock,
    rosenbrock_chained,
    extended_curved_valley,
)
from diag_ndim_noarmijo import run_no_armijo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- Quadratic test problem ----
def quadratic(n=10, kappa=10.0, seed=42):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(0.0, np.log10(kappa), n)
    A = Q @ np.diag(eigs) @ Q.T
    A = 0.5*(A + A.T)
    def f(x):
        return 0.5*float(x @ A @ x)
    def g(x):
        return A @ x
    return dict(name=f"Quadratic κ={kappa:g}",
                short=f"quad_k{int(kappa)}",
                n=n, f=f, g=g,
                x0=np.zeros(n), x_star=np.zeros(n),
                fstar=0.0)


# ---- x^* map для существующих задач ----
def _x_star(short, n):
    if short in ('ros_chain', 'ext_rosen'):
        return np.ones(n)
    if short.startswith('ecv'):
        return np.zeros(n)
    raise KeyError(short)


def main():
    rng = np.random.default_rng(20260503)
    n_dirs = 50
    max_iter = 500
    tol = 1e-8
    p_window = 5
    methods_all = ['sr1', 'ss_sr1', 'psb', 'ss_psb']
    radii = [0.3, 0.1, 0.03]

    # --- Те же 5 задач ---
    ecv_configs = [(10, 100.0, 1.5),
                   (20, 100.0, 1.5),
                   (50, 100.0, 1.5)]
    mgh_problems_n10 = ['ros_chain', 'ext_rosen']

    # Те же 50 направлений, что в основном эксперименте.
    U_by_n = {}
    needed_ns = sorted({nv for nv, _, _ in ecv_configs} | {10})
    for nv in needed_ns:
        Z = rng.standard_normal((n_dirs, nv))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        U_by_n[nv] = Z

    standard_problems = []
    for nv, a, b in ecv_configs:
        standard_problems.append((extended_curved_valley(n=nv, alpha=a, beta=b), nv))
    for short in mgh_problems_n10:
        prob = (rosenbrock_chained(10) if short == 'ros_chain'
                else extended_rosenbrock(10))
        standard_problems.append((prob, 10))

    # --- Квадратики ---
    quadratics = [quadratic(n=10, kappa=10.0),
                  quadratic(n=10, kappa=100.0)]

    summary_lines = [
        "# diag_ndim_noarmijo_local.py — БЕЗ Armijo, локальные старты",
        f"# n_dirs={n_dirs}  max_iter={max_iter}  tol={tol}  seed=20260503  p={p_window}",
        "",
    ]

    # ============================================================
    #   Часть 1: существующие 5 задач × 3 радиуса × 4 метода
    # ============================================================
    print("="*80)
    print("Часть 1: стандартные 5 задач с локальными стартами вокруг x^*")
    print("="*80)
    summary_lines.append("## Стандартные задачи: старты x^* + R·u, u ∈ S^{n-1}")
    summary_lines.append("")
    summary_lines.append(f"{'problem':22s} {'n':>3s} {'R':>5s} {'method':8s} "
                         f"{'conv':>5s} {'med_it':>7s} {'div':>4s} {'max':>4s}")
    summary_lines.append("-"*80)

    t_start = time.time()
    for prob, nv in standard_problems:
        x_star = _x_star(prob['short'], nv)
        print(f"\n[{prob['name']}]  n={nv}, x^*={x_star[:min(3,nv)].tolist()}...")
        for R in radii:
            for mk in methods_all:
                trajs = []
                t0 = time.time()
                for di in range(n_dirs):
                    x0 = x_star + R * U_by_n[nv][di]
                    tr = run_no_armijo(prob, x0, mk, p_window=p_window,
                                       max_iter=max_iter, tol=tol)
                    trajs.append(tr)
                dt = time.time() - t0
                n_conv = sum(1 for t in trajs if t['converged'])
                n_div  = sum(1 for t in trajs if t['status'] == 'diverge')
                n_max  = sum(1 for t in trajs if t['status'] == 'max_iter')
                med_iter = (int(np.median([t['iters'] for t in trajs if t['converged']]))
                            if n_conv > 0 else -1)
                print(f"  R={R:>4.2f}  [{mk:8s}] conv={n_conv:2d}/{n_dirs}"
                      f"  div={n_div:2d}  max={n_max:2d}"
                      f"  med_it={med_iter:4d}  ({dt:4.1f}s)")
                summary_lines.append(
                    f"{prob['short']:22s} {nv:>3d} {R:>5.2f} {mk:8s} "
                    f"{n_conv:>2d}/{n_dirs} {med_iter:>7d} {n_div:>4d} {n_max:>4d}")

    # ============================================================
    #   Часть 2: квадратики
    # ============================================================
    print()
    print("="*80)
    print("Часть 2: квадратики f(x) = (1/2) x^T A x, x^* = 0")
    print("="*80)
    summary_lines.append("")
    summary_lines.append("## Квадратики: x_0 = R·u (x^* = 0)")
    summary_lines.append("")
    summary_lines.append(f"{'problem':22s} {'n':>3s} {'R':>5s} {'method':8s} "
                         f"{'conv':>5s} {'med_it':>7s} {'div':>4s} {'max':>4s}")
    summary_lines.append("-"*80)

    for prob in quadratics:
        nv = prob['n']
        print(f"\n[{prob['name']}]  n={nv}")
        for R in radii:
            for mk in methods_all:
                trajs = []
                t0 = time.time()
                for di in range(n_dirs):
                    x0 = R * U_by_n[nv][di]
                    tr = run_no_armijo(prob, x0, mk, p_window=p_window,
                                       max_iter=max_iter, tol=tol)
                    trajs.append(tr)
                dt = time.time() - t0
                n_conv = sum(1 for t in trajs if t['converged'])
                n_div  = sum(1 for t in trajs if t['status'] == 'diverge')
                n_max  = sum(1 for t in trajs if t['status'] == 'max_iter')
                med_iter = (int(np.median([t['iters'] for t in trajs if t['converged']]))
                            if n_conv > 0 else -1)
                print(f"  R={R:>4.2f}  [{mk:8s}] conv={n_conv:2d}/{n_dirs}"
                      f"  div={n_div:2d}  max={n_max:2d}"
                      f"  med_it={med_iter:4d}  ({dt:4.1f}s)")
                summary_lines.append(
                    f"{prob['short']:22s} {nv:>3d} {R:>5.2f} {mk:8s} "
                    f"{n_conv:>2d}/{n_dirs} {med_iter:>7d} {n_div:>4d} {n_max:>4d}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    out = os.path.join(SCRIPT_DIR, "ndim_noarmijo_local_summary.txt")
    with open(out, "w") as fp:
        fp.write(summary + "\n")
    print(f"\nsaved: {out}  ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()
