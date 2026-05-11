"""diag_ndim_noarmijo.py — повтор экспериментов главы 3 БЕЗ Armijo.

Цель: эмпирически проверить, какую роль играет глобализация (предположение
(H4) в теореме сверхлинейной сходимости) для SR1/PSB и их SS-вариантов.
Все остальные параметры идентичны diag_ndim_stat.py (тот же seed
20260503, те же 50 направлений на S^{n-1} в paired-дизайне, p_max=5,
max_iter=500, tol=1e-8).

Отличие от diag_ndim_stat.run:
  * шаг — полный QN: x_{k+1} = x_k + d_k, где B_k d_k = -∇f(x_k)
    (никакого backtracking, никакого fallback'а на -∇f при g·d ≥ 0)
  * критерий расхождения: ||x|| > 1e10 или nan/inf
  * eigh-fail в SS-коррекции = пропуск коррекции (как и в Armijo-версии)

Выход: ndim_noarmijo_summary.txt — таблица success rate / median iters,
готовая к сравнению с ndim_stat_summary.txt.
"""
from __future__ import annotations

import os
import time
import numpy as np
from numpy.linalg import norm, solve, LinAlgError

from diag_ndim_stat import (
    extended_rosenbrock,
    rosenbrock_chained,
    extended_curved_valley,
    sr1_step,
    psb_step,
    ss_sr1_step,
    ss_psb_step,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_no_armijo(prob, x0, method, p_window=5, max_iter=500, tol=1e-8):
    """Тот же контракт, что diag_ndim_stat.run, но без Armijo: a≡1, без fallback."""
    n = prob['n']
    f, gfun = prob['f'], prob['g']
    x = x0.copy()
    B = np.eye(n)

    g = gfun(x)
    n_f, n_g = 0, 1
    hist_g = [norm(g)]
    S_buf = np.zeros((n, p_window))
    Y_buf = np.zeros((n, p_window))
    m_buf = 0
    converged = (hist_g[-1] <= tol)
    status = "converged" if converged else "max_iter"

    for k in range(max_iter):
        if hist_g[-1] <= tol:
            converged = True; status = "converged"; break

        try:
            d = solve(B, -g)
        except LinAlgError:
            status = "ls_fail"; break  # сингулярный B без Armijo — труба

        if not np.all(np.isfinite(d)):
            status = "diverge"; break

        s = d  # полный QN-шаг
        x_new = x + s
        if not np.all(np.isfinite(x_new)) or norm(x_new) > 1e10:
            status = "diverge"; break

        g_new = gfun(x_new); n_g += 1
        if not np.all(np.isfinite(g_new)):
            status = "diverge"; break
        y = g_new - g

        if m_buf < p_window:
            S_buf[:, m_buf] = s; Y_buf[:, m_buf] = y; m_buf += 1
        else:
            S_buf[:, :-1] = S_buf[:, 1:]; S_buf[:, -1] = s
            Y_buf[:, :-1] = Y_buf[:, 1:]; Y_buf[:, -1] = y

        Sw = S_buf[:, :m_buf]; Yw = Y_buf[:, :m_buf]
        if   method == 'sr1':     B = sr1_step(B, s, y)
        elif method == 'psb':     B = psb_step(B, s, y)
        elif method == 'ss_sr1':  B = ss_sr1_step(B, s, y, Sw, Yw)
        elif method == 'ss_psb':  B = ss_psb_step(B, s, y, Sw, Yw)
        else:
            raise ValueError(method)

        if not np.all(np.isfinite(B)):
            status = "diverge"; break

        x, g = x_new, g_new
        hist_g.append(norm(g))

    return dict(method=method, problem=prob['short'], n=n,
                converged=converged, status=status,
                iters=len(hist_g)-1, n_f=n_f, n_g=n_g,
                g_final=hist_g[-1])


def main():
    rng = np.random.default_rng(20260503)
    n_dirs = 50
    max_iter = 500
    tol = 1e-8
    p_window = 5

    methods_all = ['sr1', 'ss_sr1', 'psb', 'ss_psb']
    ecv_configs = [(10, 100.0, 1.5),
                   (20, 100.0, 1.5),
                   (50, 100.0, 1.5)]
    mgh_problems_n10 = ['ros_chain', 'ext_rosen']

    # Те же 50 направлений, что в diag_ndim_stat.py — paired-дизайн.
    U_by_n = {}
    needed_ns = sorted({nv for nv, _, _ in ecv_configs} | {10})
    for nv in needed_ns:
        Z = rng.standard_normal((n_dirs, nv))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        U_by_n[nv] = Z

    problems = []
    for nv, a, b in ecv_configs:
        problems.append((extended_curved_valley(n=nv, alpha=a, beta=b),
                         nv, ('ecv', nv)))
    for short in mgh_problems_n10:
        prob = (rosenbrock_chained(10) if short == 'ros_chain'
                else extended_rosenbrock(10))
        problems.append((prob, 10, (short, 10)))

    summary_lines = [
        f"# diag_ndim_noarmijo.py — БЕЗ Armijo, полный QN-шаг x_{{k+1}}=x_k+d_k",
        f"# n_dirs={n_dirs}  max_iter={max_iter}  tol={tol}  seed=20260503  p={p_window}",
        "",
        f"{'problem':22s} {'n':>3s} {'method':8s} {'conv':>5s} "
        f"{'med_iter':>9s} {'med_∇f':>8s} "
        f"{'div':>4s} {'lsf':>4s} {'max':>4s} {'R0':>8s}",
        "-"*80,
    ]

    raw = {}
    t_start = time.time()
    for prob, nv, key in problems:
        R0 = norm(prob['x0'])
        print(f"\n[{prob['name']}]  n={nv}  R0={R0:.3f}")
        for mk in methods_all:
            trajs = []
            t0 = time.time()
            for di in range(n_dirs):
                x0 = R0 * U_by_n[nv][di]
                tr = run_no_armijo(prob, x0, mk,
                                   p_window=p_window,
                                   max_iter=max_iter, tol=tol)
                trajs.append(tr)
            dt = time.time() - t0
            n_conv = sum(1 for t in trajs if t['converged'])
            n_div  = sum(1 for t in trajs if t['status'] == 'diverge')
            n_lsf  = sum(1 for t in trajs if t['status'] == 'ls_fail')
            n_max  = sum(1 for t in trajs if t['status'] == 'max_iter')
            med_iter = (int(np.median([t['iters'] for t in trajs if t['converged']]))
                        if n_conv > 0 else -1)
            med_ng = (int(np.median([t['n_g'] for t in trajs if t['converged']]))
                      if n_conv > 0 else -1)
            print(f"  [{mk:8s}] conv={n_conv:2d}/{n_dirs}"
                  f"  div={n_div:2d}  lsf={n_lsf:2d}  max={n_max:2d}"
                  f"  med_iter={med_iter:4d}  med_∇f={med_ng:4d}  ({dt:5.1f}s)")
            short_name = key[0]
            summary_lines.append(
                f"{short_name:22s} {nv:>3d} {mk:8s} {n_conv:>2d}/{n_dirs} "
                f"{med_iter:>9d} {med_ng:>8d} "
                f"{n_div:>4d} {n_lsf:>4d} {n_max:>4d} {R0:>8.3f}")
            raw[(*key, mk)] = trajs

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    out = os.path.join(SCRIPT_DIR, "ndim_noarmijo_summary.txt")
    with open(out, "w") as fp:
        fp.write(summary + "\n")
    print(f"\nsaved: {out}  ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()
