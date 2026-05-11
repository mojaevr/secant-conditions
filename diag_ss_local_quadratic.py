"""diag_ss_local_quadratic.py — графики локальной сходимости без Armijo.

Иллюстрирует Q-сверхлинейный режим теоремы thm:ss_dm на выпуклой
квадратике f(x) = (1/2) x^T A x:
  * (H4) выполнено тривиально (квадратика → полный QN-шаг минимизирует f);
  * SR1 имеет конечную сходимость за ≤ n+1 шагов;
  * SS-PSB ускоряет PSB локально (предсказание Q-сверхлинейности).

Выход: mipt_thesis_master/fig_ss_local_quadratic.pdf — 2 панели
(κ=10 и κ=100), траектории ||∇f(x_k)||: медиана + IQR по 50 стартам.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
from numpy.linalg import norm, solve, LinAlgError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diag_ndim_stat import sr1_step, psb_step, ss_sr1_step, ss_psb_step

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


def quadratic(n=10, kappa=10.0, seed=42):
    """f(x) = (1/2) x^T A x, A = Q diag(1,…,κ) Q^T, Q ∈ O(n), x* = 0.

    Чистая выпуклая квадратика — учебниковый локальный случай теоремы
    thm:ss_dm. (H4) выполнено тривиально (полный QN-шаг = шаг Ньютона
    для квадратики, и B_k обновляется так, что B_k → A).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(0.0, np.log10(kappa), n)
    A = Q @ np.diag(eigs) @ Q.T
    A = 0.5*(A + A.T)
    def f(x):
        return 0.5*float(x @ A @ x)
    def g(x):
        return A @ x
    return dict(name=f"Quadratic κ={kappa:g}", n=n, f=f, g=g)


def run_traced(prob, x0, method, p_window=5, max_iter=80, tol=1e-12):
    """run_no_armijo + возврат полной истории ||g_k||."""
    n = prob['n']
    gfun = prob['g']
    x = x0.copy()
    B = np.eye(n)
    g = gfun(x)
    hist = [norm(g)]
    S_buf = np.zeros((n, p_window))
    Y_buf = np.zeros((n, p_window))
    m_buf = 0
    for _ in range(max_iter):
        if hist[-1] <= tol:
            break
        try:
            d = solve(B, -g)
        except LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break
        s = d
        x_new = x + s
        if not np.all(np.isfinite(x_new)) or norm(x_new) > 1e10:
            break
        g_new = gfun(x_new)
        if not np.all(np.isfinite(g_new)):
            break
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
        if not np.all(np.isfinite(B)):
            break
        x, g = x_new, g_new
        hist.append(norm(g))
    return np.array(hist)


def pad_to(arrs, length):
    """Pad each trajectory to length with NaN at the tail (после сходимости)."""
    out = np.full((len(arrs), length), np.nan)
    for i, a in enumerate(arrs):
        m = min(len(a), length)
        out[i, :m] = a[:m]
    return out


def main():
    rng = np.random.default_rng(20260503)
    n_dirs = 50
    R = 0.3
    max_iter = 400
    tol = 1e-12

    panels = [
        dict(n=20, kappa=1000.0,
             title=r'$n=20$, $\kappa(A)=10^{3}$'),
        dict(n=50, kappa=100.0,
             title=r'$n=50$, $\kappa(A)=10^{2}$'),
    ]

    needed_ns = sorted({p['n'] for p in panels})
    U_by_n = {}
    for n in needed_ns:
        Z = rng.standard_normal((n_dirs, n))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        U_by_n[n] = Z

    methods = ['psb', 'ss_psb']
    styles = {
        'psb':    dict(label='PSB',    color='#d62728', ls=':'),
        'ss_psb': dict(label='SS-PSB', color='#d62728', ls='-'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    for ax, panel in zip(axes, panels):
        n = panel['n']
        prob = quadratic(n=n, kappa=panel['kappa'])
        starts = R * U_by_n[n]
        for mk in methods:
            trajs = []
            for di in range(n_dirs):
                h = run_traced(prob, starts[di], mk,
                               max_iter=max_iter, tol=tol)
                trajs.append(h)
            H = pad_to(trajs, max_iter+1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(H, axis=0)
                q25 = np.nanpercentile(H, 25, axis=0)
                q75 = np.nanpercentile(H, 75, axis=0)
            ks = np.arange(len(med))
            st = styles[mk]
            ax.plot(ks, med, color=st['color'], ls=st['ls'], lw=1.7,
                    label=st['label'])
            ax.fill_between(ks, q25, q75, color=st['color'], alpha=0.10, lw=0)
        ax.axhline(tol, color='gray', ls=':', lw=0.8, alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel(r'итерация $k$')
        ax.set_xlim(0, max_iter)
        ax.set_ylim(1e-13, None)
        ax.set_title(panel['title'], fontsize=11)
        ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    axes[0].set_ylabel(r'$\|\nabla f(x_k)\|$')
    fig.suptitle(
        r'Локальная сходимость без Armijo: '
        r'$f(x)=\frac{1}{2}\,x^\top A x$,'
        rf' $\|x_0\|={R}$, $50$ стартов;'
        r' медиана и межквартильный коридор',
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(THESIS_DIR, "fig_ss_local_quadratic.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
