"""
Численная проверка теоремы о финитном завершении SP-Broyden на линейных F.

Утверждение, которое проверяем:
  Для F(x) = Ax - b с невырожденной A, SP-Broyden с глубиной окна p+1
  достигает точности eps за K(p) итераций, где K(p) удовлетворяет

      K(p) ~ n/(p+1) + const

  под условием полного ранга кумулятивных секущих.
  У классического Бройдена (p=0) известно (Gay, 1979), что K <= 2n.

  Если empirical K(p) ложится близко к n/(p+1), теорема валидна.
  Если нет --- условие покрытия нарушается, и теорема нуждается в
  дополнительных гипотезах.

Output:
  fig_linear_finite.pdf       --- К(p) vs p, n in {50, 100, 200}
  fig_linear_finite_rank.pdf  --- кумулятивный ранг секущих
  linear_finite.npz           --- сырые данные
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve, matrix_rank
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sp_broyden_linear(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    B0: np.ndarray,
    p_max: int,
    cond_thresh: float = 1e8,
    tol: float = 1e-12,
    maxiter: int = 1500,
    track_rank: bool = True,
):
    """SP-Broyden для F(x) = Ax - b. Возвращает словарь с диагностикой."""
    n = len(x0)
    x = x0.copy().astype(float)
    Fx = A @ x - b
    B = B0.copy().astype(float)

    out = {
        "res": [float(norm(Fx))],
        "p_eff": [0],
        "cond_Sp": [0.0],
        "cum_rank": [0],
    }
    S_all = []
    S_hist, Y_hist = [], []

    for k in range(maxiter):
        if norm(Fx) < tol:
            break
        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break

        x_new = x + d
        Fx_new = A @ x_new - b
        if not np.all(np.isfinite(Fx_new)):
            break

        y = Fx_new - Fx  # Для линейной F это в точности A @ d, но считаем явно
        s = d.copy()
        S_hist.append(s.copy())
        Y_hist.append(y.copy())
        S_all.append(s.copy())

        # --- адаптивный выбор p ---
        Bs = B @ s
        p_eff = 0
        cond_used = 1.0
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                cG = cond(G)
                if cG < cond_thresh:
                    p_eff = p_try
                    cond_used = float(cG)
                else:
                    break

        if p_eff == 0:
            v = s
        else:
            cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols)
            G = Sp.T @ Sp
            e1 = np.zeros(p_eff + 1)
            e1[0] = 1.0
            try:
                v = Sp @ solve(G, e1)
            except np.linalg.LinAlgError:
                v = s
                p_eff = 0

        denom = float(v @ s)
        if abs(denom) < 1e-14:
            v = s
            denom = float(s @ s)

        # --- обновление Бройдена/SP-Broyden ---
        # Замечание: при p_eff > 0 формула B + (y - Bs) v^T / (v^T s) с v = Sp G^{-1} e1
        # даёт классический мульти-секущий апдейт (доказательство --- lem:equivalence).
        B = B + np.outer(y - Bs, v) / denom

        x = x_new
        Fx = Fx_new
        out["res"].append(float(norm(Fx)))
        out["p_eff"].append(int(p_eff))
        out["cond_Sp"].append(cond_used)
        if track_rank:
            S_mat = np.column_stack(S_all)
            out["cum_rank"].append(int(matrix_rank(S_mat, tol=1e-10)))
        else:
            out["cum_rank"].append(0)

        # скользящее окно
        max_hist = max(p_max + 5, 25)
        if len(S_hist) > max_hist:
            S_hist.pop(0)
            Y_hist.pop(0)

    return out


def make_random_linear_system(n: int, kappa: float, rng: np.random.Generator):
    """A с заданным cond, b случайный, x* = A^-1 b, x_0 = 0."""
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sigmas = np.geomspace(1.0, 1.0 / kappa, n)
    A = Q1 @ np.diag(sigmas) @ Q2.T
    x_star = rng.standard_normal(n)
    b = A @ x_star
    x0 = np.zeros(n)
    return A, b, x0, x_star


# ----------------------------------------------------------------------
# Основной эксперимент.
# ----------------------------------------------------------------------


def main():
    SEED = 20260502
    NS = [50, 100, 200]
    PS = [0, 1, 2, 5, 10, 20]
    KAPPA = 5.0  # умеренное обусловление: A не близка к вырожденной
    REPLICATES = 5
    TOL = 1e-12
    MAXITER = 2500

    OUTDIR = os.path.join(os.path.dirname(__file__), "mipt_thesis_master")

    results = {}  # (n, p) -> list of K (по replicates)
    rank_traces = {}  # (n, p) -> last rank trajectory (for one replicate)
    res_traces = {}  # (n, p) -> last residual trajectory (for one replicate)

    for n in NS:
        for p in PS:
            Ks = []
            for r in range(REPLICATES):
                rng = np.random.default_rng(SEED + 1000 * NS.index(n) + 17 * p + r)
                A, b, x0, x_star = make_random_linear_system(n, KAPPA, rng)
                B0 = np.eye(n)
                out = sp_broyden_linear(
                    A, b, x0, B0, p_max=p, tol=TOL, maxiter=MAXITER
                )
                K = len(out["res"]) - 1  # количество итераций
                converged = out["res"][-1] < TOL
                Ks.append(K if converged else MAXITER)
                if r == REPLICATES - 1:
                    rank_traces[(n, p)] = out["cum_rank"]
                    res_traces[(n, p)] = out["res"]
            results[(n, p)] = Ks
            print(
                f"n={n:3d}, p={p:2d}: K = {np.median(Ks):6.1f} "
                f"(min {min(Ks)}, max {max(Ks)}, replicates {REPLICATES}), "
                f"theory n/(p+1)={n/(p+1):.1f}"
            )

    # -----------------------------------------------------------------
    # Фигура 1: K(p) vs theoretical n/(p+1)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, n in zip(axes, NS):
        meds = [float(np.median(results[(n, p)])) for p in PS]
        mins = [float(np.min(results[(n, p)])) for p in PS]
        maxs = [float(np.max(results[(n, p)])) for p in PS]
        theory = [n / (p + 1) for p in PS]
        ax.plot(PS, theory, "k--", label=r"теория: $n/(p+1)$", linewidth=1.5)
        ax.errorbar(
            PS,
            meds,
            yerr=[np.array(meds) - np.array(mins), np.array(maxs) - np.array(meds)],
            fmt="o-",
            color="tab:blue",
            label="медиана и min/max",
            capsize=3,
        )
        # Бройден reference: 2n (Gay 1979)
        ax.axhline(2 * n, color="tab:red", linestyle=":", alpha=0.7,
                   label=r"граница Бройдена $2n$ (Gay)")
        ax.set_xlabel(r"$p$ (глубина окна $-1$)")
        ax.set_ylabel("итераций до $\|F\|<10^{-12}$")
        ax.set_title(f"$n={n}$, $\kappa(A)={KAPPA}$")
        ax.set_yscale("log")
        ax.set_xticks(PS)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_linear_finite.pdf"))
    plt.close(fig)

    # -----------------------------------------------------------------
    # Фигура 2: ранг кумулятивных секущих + residual для n=100
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n_show = 100
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(PS)))
    for ax in axes:
        ax.set_xlabel("итерация $k$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"rank$([s_0,\ldots,s_{k-1}])$")
    axes[0].set_title(f"Кумулятивный ранг секущих ($n={n_show}$)")
    axes[0].axhline(n_show, color="k", linestyle=":", alpha=0.5,
                    label=f"$n={n_show}$")
    for c, p in zip(cmap, PS):
        rt = rank_traces[(n_show, p)]
        axes[0].plot(rt, color=c, label=f"$p={p}$")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_ylabel(r"$\|F(x_k)\|_2$")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Невязка ($n={n_show}$)")
    for c, p in zip(cmap, PS):
        rt = res_traces[(n_show, p)]
        axes[1].plot(rt, color=c, label=f"$p={p}$")
    axes[1].legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_linear_finite_rank.pdf"))
    plt.close(fig)

    # -----------------------------------------------------------------
    # Сохраняем сырые данные
    # -----------------------------------------------------------------
    np.savez(
        os.path.join(os.path.dirname(__file__), "linear_finite.npz"),
        ns=np.array(NS),
        ps=np.array(PS),
        kappa=KAPPA,
        replicates=REPLICATES,
        K_median=np.array([
            [float(np.median(results[(n, p)])) for p in PS] for n in NS
        ]),
        K_min=np.array([
            [float(np.min(results[(n, p)])) for p in PS] for n in NS
        ]),
        K_max=np.array([
            [float(np.max(results[(n, p)])) for p in PS] for n in NS
        ]),
    )
    print("\nДанные --> linear_finite.npz")
    print(f"Фигуры --> {OUTDIR}/fig_linear_finite{{,_rank}}.pdf")

    # -----------------------------------------------------------------
    # Регрессия: K(p) ~ a * n/(p+1) + b
    # -----------------------------------------------------------------
    print("\nРегрессия K(p) = a * n/(p+1) + b на медианах:")
    for n in NS:
        meds = np.array([float(np.median(results[(n, p)])) for p in PS])
        x_reg = np.array([n / (p + 1) for p in PS])
        a, b = np.polyfit(x_reg, meds, 1)
        residual = float(np.linalg.norm(meds - (a * x_reg + b)))
        print(
            f"  n={n}: a={a:.3f}, b={b:.2f}, |residual|={residual:.2f}, "
            f"теория a=1, b<<n"
        )


if __name__ == "__main__":
    main()
