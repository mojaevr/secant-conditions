"""
diag_highdim_stat.py — статистически корректная замена fig:highdim_conv.

Задача: Broyden Banded (MGH #31) при n ∈ {10^3, 10^4}.
Цель: показать (i) сходимость limited-memory L-SP-Broyden при m ∈ {2,5,10,20},
(ii) приближение к dense SP-Broyden-SM при m ↑ (только n=10^3, dense невозможен
при n=10^4).

10 случайных стартов: x_0 = -0.1·1 + 0.05·u, u ~ N(0, I/n)/||u||.
Все методы — с глобализацией Армихо (Алгоритм 2). Медиана + IQR по стартам.

Выход: mipt_thesis_master/fig_highdim_conv.pdf (перезаписывает старый
4-панельный график).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diag_highdim as dh   # импорт реализаций методов


def main():
    # Настройки для трёх размерностей. dense_seeds — сколько прогонов dense
    # позволяем себе (на n=10^4 он стоит ~800 MB и заметное wall-time).
    configs = [
        dict(n=10_000, n_seeds=8, dense_seeds=3, m_values=[5, 10, 20]),
        dict(n=100_000, n_seeds=4, dense_seeds=0, m_values=[10, 20]),
    ]

    rng = np.random.default_rng(20260502)

    fig, axes = plt.subplots(1, len(configs), figsize=(11.0, 4.2), sharey=False)

    for idx, cfg in enumerate(configs):
        n = cfg["n"]; n_seeds = cfg["n_seeds"]
        M_VALUES = cfg["m_values"]; dense_seeds = cfg["dense_seeds"]
        ax = axes[idx]
        x0_default = dh.broyden_banded_x0(n)
        dirs = rng.standard_normal((n_seeds, n))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        starts = [x0_default + 0.05 * u for u in dirs]

        cmap = plt.get_cmap("viridis")
        m_colors = {m: cmap(i / max(1, len(M_VALUES) - 1))
                    for i, m in enumerate(M_VALUES)}

        # --- L-SP-Broyden, разные m ---
        for m in M_VALUES:
            trajs = []
            for x0 in starts:
                r = dh.lsp_broyden(dh.broyden_banded_F, x0, m=m, p_max=5,
                                    maxiter=300, tol=1e-10, globalize=True)
                trajs.append(r["res"])
            L = max(len(t) for t in trajs)
            M = np.full((n_seeds, L), np.nan)
            for i, t in enumerate(trajs):
                M[i, :len(t)] = t
            med = np.nanmedian(M, axis=0)
            q25 = np.nanquantile(M, 0.25, axis=0)
            q75 = np.nanquantile(M, 0.75, axis=0)
            valid = (~np.isnan(M)).sum(axis=0) >= max(1, n_seeds // 2)
            last = (np.where(valid)[0][-1] + 1) if valid.any() else 0
            sl = slice(0, last)
            ks = np.arange(L)
            color = m_colors[m]
            ax.fill_between(ks[sl], q25[sl], q75[sl], color=color, alpha=0.15, lw=0)
            ax.semilogy(ks[sl], med[sl], color=color, lw=1.7,
                        label=fr"L-SP-Broyden, $m={m}$")
            print(f"  n={n}, m={m}: median iters = "
                  f"{int(np.median([len(t) for t in trajs]))}")

        # --- dense SP-Broyden-SM (только если dense_seeds>0) ---
        if dense_seeds > 0:
            trajs_dense = []
            for x0 in starts[:dense_seeds]:
                r = dh.sp_broyden_sm(dh.broyden_banded_F, x0, p_max=5,
                                      maxiter=300, tol=1e-10, globalize=True)
                trajs_dense.append(r["res"])
            L = max(len(t) for t in trajs_dense)
            M = np.full((dense_seeds, L), np.nan)
            for i, t in enumerate(trajs_dense):
                M[i, :len(t)] = t
            med = np.nanmedian(M, axis=0)
            q25 = np.nanquantile(M, 0.25, axis=0)
            q75 = np.nanquantile(M, 0.75, axis=0)
            valid = (~np.isnan(M)).sum(axis=0) >= 1
            last = (np.where(valid)[0][-1] + 1) if valid.any() else 0
            sl = slice(0, last)
            ks = np.arange(L)
            ax.fill_between(ks[sl], q25[sl], q75[sl], color="#000000", alpha=0.10, lw=0)
            ax.semilogy(ks[sl], med[sl], color="#000000", lw=2.0, ls=(0, (3, 2)),
                        label=r"dense SP-Broyden-SM ($p\leq 5$)")
            print(f"  n={n}, dense ({dense_seeds} стартов): median iters = "
                  f"{int(np.median([len(t) for t in trajs_dense]))}")

        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(r"$\|F(x_k)\|_2$")
        title_suffix = ""
        if dense_seeds == 0:
            title_suffix = "  (dense > 80 ГБ, невозможен)"
        ax.set_title(fr"Broyden Banded, $n=10^{int(np.log10(n))}$" + title_suffix)
        ax.axhline(1e-10, color="#888", lw=0.6, ls=":")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mipt_thesis_master", "fig_highdim_conv.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()
