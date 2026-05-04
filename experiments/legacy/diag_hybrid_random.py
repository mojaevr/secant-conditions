"""
Гибрид: случайные линейные комбинации исторических секущих в SP-Broyden.

Идея: на шаге k, вместо траекторных rolling-секущих или свежих гауссовских
зондов, использовать p+1 случайных линейных комбинаций уже накопленной
истории {s_j, y_j}_{j=0}^{k-1}.

  c_i ~ N(0, I/m), v_i = S_hist @ c_i, z_i = Y_hist @ c_i  (i=1,...,p+1)

По линейности z_i = A v_i для линейной F (никаких новых F-вызовов).
Это "sketch внутри range(S_hist)": алгоритм сжимает E_k только на
подпространстве истории, со скоростью (p+1)/d_k, где d_k = rank(history).

Сравниваем 4 варианта на той же задаче:
  - Broyden                  (p=0, rolling)
  - SP-Broyden rolling       (p>0, последние p+1 секущих)
  - SP-Broyden fresh sketch  (p>0, p+1 свежих гауссовских зондов; не free)
  - SP-Broyden hybrid        (p>0, p+1 случайных комбинаций истории; free)

Output:
  fig_hybrid_random.pdf   --- сравнение четырёх вариантов
  hybrid_random.npz       --- сырые данные
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve, matrix_rank
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sp_broyden_unified(
    A, b, x0, B0, p_max, mode,
    cond_thresh=1e8, tol=1e-12, maxiter=1500, rng=None,
    history_window=None,  # для hybrid: ограничение длины истории; None=всё
):
    """SP-Broyden с четырьмя режимами выбора секущих:
      'rolling'       : последние p+1 траекторных секущих
      'fresh_sketch'  : p+1 свежих гауссовских зондов, y = A v (1 матвекс)
      'hybrid_random' : p+1 случайных комбинаций истории (free for linear F)
      'broyden'       : p=0, классический Бройден
    """
    n = len(x0)
    x = x0.copy().astype(float)
    Fx = A @ x - b
    B = B0.copy().astype(float)
    out = {
        "res": [float(norm(Fx))],
        "jac_err": [float(norm(B - A, "fro"))],
        "n_extra_matvecs": 0,  # сколько ДОПОЛНИТЕЛЬНЫХ A·v потребовалось
    }
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
        y = Fx_new - Fx
        s = d.copy()
        S_hist.append(s.copy())
        Y_hist.append(y.copy())

        # === Выбор Π_k ===
        if mode == "broyden":
            Bs = B @ s
            v = s
            denom = float(s @ s)
            B = B + np.outer(y - Bs, v) / denom

        elif mode == "rolling":
            Bs = B @ s
            p_eff = 0
            if p_max > 0 and len(S_hist) >= 2:
                for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                    cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                    Sp = np.column_stack(cols)
                    G = Sp.T @ Sp
                    cG = cond(G)
                    if cG < cond_thresh:
                        p_eff = p_try
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
                v = Sp @ solve(G, e1)
            denom = float(v @ s)
            if abs(denom) < 1e-14:
                v = s
                denom = float(s @ s)
            B = B + np.outer(y - Bs, v) / denom

        elif mode == "fresh_sketch":
            # p+1 случайных гауссовских зондов; y = A v (extra matvecs!)
            S_sk = rng.standard_normal((n, p_max + 1))
            Y_sk = A @ S_sk
            out["n_extra_matvecs"] += p_max + 1
            BS = B @ S_sk
            G = S_sk.T @ S_sk
            try:
                B = B + (Y_sk - BS) @ solve(G, S_sk.T)
            except np.linalg.LinAlgError:
                pass

        elif mode == "hybrid_random":
            # p+1 случайных комбинаций истории --- free
            m = len(S_hist)
            if history_window is not None:
                m = min(m, history_window)
                S_use = np.column_stack(S_hist[-m:])
                Y_use = np.column_stack(Y_hist[-m:])
            else:
                S_use = np.column_stack(S_hist)
                Y_use = np.column_stack(Y_hist)
            if m < p_max + 1:
                # история короче окна --- используем всё, что есть
                # эквивалент rolling с эфф. p
                p_eff_h = m - 1
                if p_eff_h < 0:
                    p_eff_h = 0
                cols = list(range(m))  # всё, что есть
                Sp = S_use
                G = Sp.T @ Sp
                Bs = B @ s
                if p_eff_h == 0:
                    v = s
                else:
                    e1 = np.zeros(m); e1[0] = 1.0
                    try:
                        v = Sp @ solve(G, e1)
                    except np.linalg.LinAlgError:
                        v = s
                denom = float(v @ s)
                if abs(denom) < 1e-14:
                    v = s; denom = float(s @ s)
                B = B + np.outer(y - Bs, v) / denom
            else:
                # генерируем p+1 случайных комбинаций
                C = rng.standard_normal((m, p_max + 1)) / np.sqrt(m)
                V = S_use @ C  # (n, p+1)
                Z = Y_use @ C
                BV = B @ V
                G = V.T @ V
                try:
                    B = B + (Z - BV) @ solve(G, V.T)
                except np.linalg.LinAlgError:
                    pass

        else:
            raise ValueError(f"unknown mode: {mode}")

        x = x_new
        Fx = Fx_new
        out["res"].append(float(norm(Fx)))
        out["jac_err"].append(float(norm(B - A, "fro")))

        # ограничение истории
        max_hist = max(p_max + 5, 100)
        if len(S_hist) > max_hist:
            S_hist.pop(0)
            Y_hist.pop(0)

    out["K"] = len(out["res"]) - 1
    out["converged"] = out["res"][-1] < tol
    return out


def make_random_linear_system(n, kappa, rng):
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sigmas = np.geomspace(1.0, 1.0 / kappa, n)
    A = Q1 @ np.diag(sigmas) @ Q2.T
    x_star = rng.standard_normal(n)
    b = A @ x_star
    return A, b, np.zeros(n), x_star


def main():
    OUTDIR = os.path.join(os.path.dirname(__file__), "mipt_thesis_master")
    SEED = 20260502
    N = 100
    KAPPA = 5.0
    PS = [2, 5, 10, 20]
    REPLICATES = 5
    TOL = 1e-12
    MAXITER = 2500

    print(f"Гибрид: случайные комбинации истории, n={N}, k(A)={KAPPA}")
    print(f"  Бесплатно для линейной F (никаких новых матвексов)")
    print()

    results = {}
    traces_jac = {}
    traces_res = {}

    # Бройден
    Ks_br = []
    for r in range(REPLICATES):
        rng = np.random.default_rng(SEED + r)
        A, b, x0, _ = make_random_linear_system(N, KAPPA, rng)
        out = sp_broyden_unified(
            A, b, x0, np.eye(N), p_max=0, mode="broyden",
            tol=TOL, maxiter=MAXITER, rng=rng,
        )
        Ks_br.append(out["K"] if out["converged"] else MAXITER)
        if r == REPLICATES - 1:
            traces_jac[("broyden", 0)] = out["jac_err"]
            traces_res[("broyden", 0)] = out["res"]
    K_br = float(np.median(Ks_br))
    print(f"Бройден (p=0): K = {K_br:.0f}")
    results[("broyden", 0)] = Ks_br

    print(f"\n{'p':>4s} | {'rolling':>9s} | {'fresh_sk':>9s} | "
          f"{'hybrid':>9s} | {'extra_mv (fresh)':>18s}")
    print("-" * 65)
    for p in PS:
        for mode in ["rolling", "fresh_sketch", "hybrid_random"]:
            Ks = []
            extra_mv = []
            for r in range(REPLICATES):
                rng = np.random.default_rng(SEED + 100 * p + r + 17 * hash(mode) % 10000)
                A, b, x0, _ = make_random_linear_system(N, KAPPA, rng)
                out = sp_broyden_unified(
                    A, b, x0, np.eye(N), p_max=p, mode=mode,
                    tol=TOL, maxiter=MAXITER, rng=rng,
                )
                Ks.append(out["K"] if out["converged"] else MAXITER)
                extra_mv.append(out["n_extra_matvecs"])
                if r == REPLICATES - 1:
                    traces_jac[(mode, p)] = out["jac_err"]
                    traces_res[(mode, p)] = out["res"]
            results[(mode, p)] = Ks
        K_roll = float(np.median(results[("rolling", p)]))
        K_fresh = float(np.median(results[("fresh_sketch", p)]))
        K_hybrid = float(np.median(results[("hybrid_random", p)]))
        emv_fresh = float(np.median(
            [out_emv for out_emv in [results[("fresh_sketch", p)][i] for i in range(REPLICATES)]]
        ))  # просто иллюстративно
        # Точное число extra matvecs:
        emv_fresh_count = (p + 1) * K_fresh  # каждый шаг добавляет p+1 матвексов
        print(f"{p:>4d} | {K_roll:>9.0f} | {K_fresh:>9.0f} | {K_hybrid:>9.0f} | "
              f"{emv_fresh_count:>18.0f}")

    # ============= Фигура =============
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Левая: K(p)
    ax = axes[0]
    for mode, marker, color, label in [
        ("rolling",       "o", "tab:blue",  "rolling-window (детерм., free)"),
        ("fresh_sketch",  "s", "tab:red",   "fresh sketch ($+(p+1)$ матвексов/шаг)"),
        ("hybrid_random", "D", "tab:green", "hybrid random hist. (free)"),
    ]:
        meds = [float(np.median(results[(mode, p)])) for p in PS]
        mins = [float(np.min(results[(mode, p)])) for p in PS]
        maxs = [float(np.max(results[(mode, p)])) for p in PS]
        ax.errorbar(
            PS, meds,
            yerr=[np.array(meds) - np.array(mins), np.array(maxs) - np.array(meds)],
            fmt=marker + "-", color=color, label=label, capsize=3,
        )
    ax.axhline(K_br, color="k", linestyle=":", alpha=0.7,
               label=f"Бройден: {K_br:.0f}")
    ax.set_xlabel("p")
    ax.set_ylabel(r"итераций до $\|F\| < 10^{-12}$")
    ax.set_title(f"Сравнение трёх SP-стратегий (n={N}, $\\kappa(A)={KAPPA}$)")
    ax.set_xticks(PS)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Правая: ||B - A||_F траектория для p=10
    ax = axes[1]
    p_show = 10
    ax.semilogy(traces_jac[("broyden", 0)], color="k", linestyle=":",
                label="Бройден (p=0)", alpha=0.7)
    for mode, color, label in [
        ("rolling",       "tab:blue",  "rolling (free)"),
        ("fresh_sketch",  "tab:red",   "fresh sketch (+(p+1) матв/шаг)"),
        ("hybrid_random", "tab:green", "hybrid random hist. (free)"),
    ]:
        ax.semilogy(traces_jac[(mode, p_show)], color=color, label=label)
    # теория для fresh_sketch
    err0 = traces_jac[("fresh_sketch", p_show)][0]
    rate = (1 - (p_show + 1) / N)
    ks = np.arange(len(traces_jac[("fresh_sketch", p_show)]))
    ax.semilogy(ks, err0 * rate ** ks, "--",
                color="tab:red", alpha=0.4,
                label=r"теория $(1-\frac{p+1}{n})^k$ для fresh")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|B_k - A\|_F$")
    ax.set_title(f"Точность аппроксимации якобиана ($p={p_show}$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_hybrid_random.pdf"))
    plt.close(fig)

    np.savez(
        os.path.join(os.path.dirname(__file__), "hybrid_random.npz"),
        n=N, kappa=KAPPA, ps=np.array(PS), replicates=REPLICATES,
    )
    print(f"\nДанные  --> hybrid_random.npz")
    print(f"Фигура --> {OUTDIR}/fig_hybrid_random.pdf")

    print("\n=== Сводно: ускорение vs Бройдена и стоимость ===")
    print(f"{'p':>4s} | {'rolling':>10s} | {'fresh':>10s} | {'hybrid':>10s} | "
          f"{'extra mv ratio':>16s}")
    print("-" * 65)
    for p in PS:
        K_r = float(np.median(results[("rolling", p)]))
        K_f = float(np.median(results[("fresh_sketch", p)]))
        K_h = float(np.median(results[("hybrid_random", p)]))
        # стоимость fresh_sketch в матвексах: K_f * (1 + p+1) = K_f * (p+2)
        # стоимость rolling/hybrid: K * 1 = K
        # эффективная "удельная" эффективность fresh: K_f * (p+2) на ту же точность
        cost_ratio = K_f * (p + 2) / K_r  # vs rolling в матвексах
        print(f"{p:>4d} | {K_br/K_r:>9.2f}x | {K_br/K_f:>9.2f}x | {K_br/K_h:>9.2f}x | "
              f"{cost_ratio:>15.2f}x")


if __name__ == "__main__":
    main()
