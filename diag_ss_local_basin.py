"""diag_ss_local_basin.py — расширение локального бассейна сходимости.

Без Armijo (полный QN-шаг x_{k+1}=x_k+d_k) доля сходимостей на Rosenbrock
chained, n=10, в зависимости от радиуса старта R вокруг x^*=1. Старты:
x_0 = x^* + R·u, u ∈ S^{n-1}, 50 направлений (тот же seed 20260503).
Сравниваются 4 метода: SR1, SS-SR1, PSB, SS-PSB.

Цель: эмпирически показать, что SS-коррекция расширяет область
устойчивой сходимости (для PSB-семейства разница наиболее выражена).

Выход: mipt_thesis_master/fig_ss_local_basin.pdf
"""
from __future__ import annotations

import os
import warnings
import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diag_ndim_stat import rosenbrock_chained
from diag_ndim_noarmijo import run_no_armijo

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


def main():
    rng = np.random.default_rng(20260503)
    n = 10
    n_dirs = 50
    max_iter = 500
    tol = 1e-8
    p_window = 5

    radii = [1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01]
    methods = ['sr1', 'ss_sr1', 'psb', 'ss_psb']
    styles = {
        'sr1':    dict(label='SR1',    color='#1f77b4', ls=':',  marker='o'),
        'ss_sr1': dict(label='SS-SR1', color='#1f77b4', ls='-',  marker='o'),
        'psb':    dict(label='PSB',    color='#d62728', ls=':',  marker='s'),
        'ss_psb': dict(label='SS-PSB', color='#d62728', ls='-',  marker='s'),
    }

    prob = rosenbrock_chained(n=n)
    x_star = np.ones(n)
    Z = rng.standard_normal((n_dirs, n))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    success = {mk: [] for mk in methods}
    for R in radii:
        print(f"\nR={R}")
        for mk in methods:
            n_conv = 0
            for di in range(n_dirs):
                x0 = x_star + R * Z[di]
                tr = run_no_armijo(prob, x0, mk, p_window=p_window,
                                   max_iter=max_iter, tol=tol)
                if tr['converged']:
                    n_conv += 1
            success[mk].append(n_conv / n_dirs)
            print(f"  [{mk:8s}] {n_conv:2d}/{n_dirs} = {n_conv/n_dirs:.2f}")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for mk in methods:
        st = styles[mk]
        ax.plot(radii, success[mk], color=st['color'], ls=st['ls'], lw=1.7,
                marker=st['marker'], ms=5.5, label=st['label'])
    ax.set_xscale('log')
    ax.set_xlabel(r'радиус старта $R = \|x_0 - x^*\|$')
    ax.set_ylabel(r'доля сошедшихся стартов (из $50$)')
    ax.set_ylim(-0.04, 1.04)
    ax.invert_xaxis()
    ax.set_title(r'Rosenbrock chained, $n=10$,'
                 r' без Armijo ($\alpha_k\equiv 1$)', fontsize=11)
    ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(THESIS_DIR, "fig_ss_local_basin.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
