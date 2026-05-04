"""
Block-restart SP-Broyden: промежуточный детерминированный вариант
между rolling-window (полный overlap) и random sketch (без overlap).

Идея: каждые p+1 итераций *сбрасываем окно секущих* (но сохраняем B!).
Это убирает overlap между соседними блоками: блок m использует
секущие [s_{m(p+1)}, ..., s_{(m+1)(p+1)-1}], блок m+1 начинает с пустого окна.

Сравниваем три алгоритма на одной линейной задаче:
  - Broyden (p=0)
  - SP-Broyden rolling-window (p=p_max)
  - SP-Broyden block-restart (p=p_max)
  - Randomized sketch (p=p_max)

Гипотеза: если overlap rolling-window --- единственная причина
потери (p+1)x, то block-restart должен дать ~ (p+1)x. Если block-restart
тоже даёт ~ 2x, значит overlap не единственная причина и нужна
рандомизация.

Output:
  fig_block_restart.pdf  --- сравнение четырёх вариантов
  block_restart.npz      --- сырые данные
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sp_broyden_unified(
    A, b, x0, B0, p_max,
    mode="rolling",  # "rolling" | "block_restart" | "random_sketch"
    cond_thresh=1e8,
    tol=1e-12,
    maxiter=1500,
    rng=None,
):
    """SP-Broyden для F(x)=Ax-b с тремя режимами выбора окна.

    rolling     : window = последние p+1 секущих (стандартный SP-Broyden)
    block_restart: window сбрасывается каждые p+1 итераций
    random_sketch: на каждом шаге p+1 свежих гауссовских направлений,
                   y_i = A v_i (для линейной F --- 1 матвекс)
    """
    n = len(x0)
    x = x0.copy().astype(float)
    Fx = A @ x - b
    B = B0.copy().astype(float)

    out = {
        "res": [float(norm(Fx))],
        "jac_err": [float(norm(B - A, "fro"))],
        "p_eff": [0],
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

        # ===== Выбор Π_k =====
        if mode == "random_sketch":
            # Полностью игнорируем траекторные секущие; используем
            # p+1 случайных гауссовских v_i и y_i = A v_i.
            S_sketch = rng.standard_normal((n, p_max + 1))
            Y_sketch = A @ S_sketch
            BS = B @ S_sketch
            G = S_sketch.T @ S_sketch
            try:
                B = B + (Y_sketch - BS) @ solve(G, S_sketch.T)
                p_eff = p_max
            except np.linalg.LinAlgError:
                pass
                p_eff = 0
        else:  # rolling | block_restart
            # Сначала решаем, сколько секущих использовать
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

            # Стандартное SP-обновление через v
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
            B = B + np.outer(y - Bs, v) / denom

        x = x_new
        Fx = Fx_new
        out["res"].append(float(norm(Fx)))
        out["jac_err"].append(float(norm(B - A, "fro")))
        out["p_eff"].append(int(p_eff))

        # === ограничение / сброс истории ===
        if mode == "rolling":
            max_hist = max(p_max + 5, 25)
            if len(S_hist) > max_hist:
                S_hist.pop(0)
                Y_hist.pop(0)
        elif mode == "block_restart":
            # завершили блок? (k+1) итераций сделано после старта блока
            # k --- 0-indexed, стартовый k блока: если (k+1) % (p+1) == 0,
            # то после этой итерации был последний в блоке -> сброс
            if p_max > 0 and (k + 1) % (p_max + 1) == 0:
                S_hist.clear()
                Y_hist.clear()
        # random_sketch не использует S_hist для апдейта; чистим, чтобы
        # не разрасталась
        elif mode == "random_sketch":
            S_hist.clear()
            Y_hist.clear()

    return out


def make_well_conditioned_A(n, kappa, rng):
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

    print(f"Block-restart vs rolling vs random sketch на n={N}, k(A)={KAPPA}\n")

    results = {}  # (p, mode) -> list of K
    traces = {}   # (p, mode) -> jac_err trajectory (last replicate)

    MODES = ["rolling", "block_restart", "random_sketch"]
    # Бройден отдельно (p=0)
    print("=== Бройден (p=0) ===")
    Ks = []
    for r in range(REPLICATES):
        rng = np.random.default_rng(SEED + r)
        A, b, x0, x_star = make_well_conditioned_A(N, KAPPA, rng)
        out = sp_broyden_unified(
            A, b, x0, np.eye(N), p_max=0,
            mode="rolling", tol=TOL, maxiter=MAXITER, rng=rng,
        )
        K = len(out["res"]) - 1
        converged = out["res"][-1] < TOL
        Ks.append(K if converged else MAXITER)
        if r == REPLICATES - 1:
            traces[(0, "broyden")] = out["jac_err"]
    print(f"  K = {np.median(Ks):.1f} (min {min(Ks)}, max {max(Ks)})")
    results[(0, "broyden")] = Ks

    for p in PS:
        print(f"\n=== p = {p} ===")
        for mode in MODES:
            Ks = []
            for r in range(REPLICATES):
                rng = np.random.default_rng(SEED + 100 * p + r)
                A, b, x0, x_star = make_well_conditioned_A(N, KAPPA, rng)
                out = sp_broyden_unified(
                    A, b, x0, np.eye(N), p_max=p,
                    mode=mode, tol=TOL, maxiter=MAXITER, rng=rng,
                )
                K = len(out["res"]) - 1
                converged = out["res"][-1] < TOL
                Ks.append(K if converged else MAXITER)
                if r == REPLICATES - 1:
                    traces[(p, mode)] = out["jac_err"]
            results[(p, mode)] = Ks
            print(
                f"  {mode:15s}: K = {np.median(Ks):6.1f} "
                f"(min {min(Ks)}, max {max(Ks)})  "
                f"theory n/(p+1) = {N/(p+1):.1f}"
            )

    # ========= Фигура: K(p) для трёх режимов =========
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Левая панель: K(p)
    ax = axes[0]
    for mode, marker, color in zip(
        MODES, ["o", "s", "^"],
        ["tab:blue", "tab:green", "tab:red"]
    ):
        meds = [float(np.median(results[(p, mode)])) for p in PS]
        mins = [float(np.min(results[(p, mode)])) for p in PS]
        maxs = [float(np.max(results[(p, mode)])) for p in PS]
        label = {"rolling": "rolling-window (детерм.)",
                 "block_restart": "block-restart (детерм.)",
                 "random_sketch": "random sketch"}[mode]
        ax.errorbar(
            PS, meds,
            yerr=[np.array(meds) - np.array(mins),
                  np.array(maxs) - np.array(meds)],
            fmt=marker + "-", color=color, label=label, capsize=3,
        )
    # Бройден
    K_br = float(np.median(results[(0, "broyden")]))
    ax.axhline(K_br, color="k", linestyle=":", alpha=0.7,
               label=f"Бройден (p=0): {K_br:.0f}")
    # Теория
    PS_dense = np.linspace(PS[0], PS[-1], 50)
    ax.plot(PS_dense, N / (PS_dense + 1), "k--", alpha=0.5,
            label=r"теория: $n/(p+1)$")

    ax.set_xlabel("p")
    ax.set_ylabel(r"итераций до $\|F\| < 10^{-12}$")
    ax.set_title(f"Сравнение трёх вариантов (n={N}, $\\kappa(A)={KAPPA}$)")
    ax.set_yscale("log")
    ax.set_xticks(PS)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Правая панель: ||B_k - A||_F для p=10
    ax = axes[1]
    p_show = 10
    for mode, color in zip(
        MODES, ["tab:blue", "tab:green", "tab:red"]
    ):
        trace = traces[(p_show, mode)]
        label = {"rolling": "rolling-window",
                 "block_restart": "block-restart",
                 "random_sketch": "random sketch"}[mode]
        ax.semilogy(trace, color=color, label=label)
    # Бройден
    ax.semilogy(traces[(0, "broyden")], color="k", linestyle=":",
                label="Бройден (p=0)", alpha=0.7)
    # теория для random sketch
    err0 = traces[(p_show, "random_sketch")][0]
    rate = (1 - (p_show + 1) / N)
    ks = np.arange(len(traces[(p_show, "random_sketch")]))
    ax.semilogy(ks, err0 * rate ** ks,
                color="tab:red", linestyle="--", alpha=0.4,
                label=r"теория $(1-\frac{p+1}{n})^k$")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|B_k - A\|_F$")
    ax.set_title(f"Точность аппроксимации якобиана ($p={p_show}$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_block_restart.pdf"))
    plt.close(fig)

    np.savez(
        os.path.join(os.path.dirname(__file__), "block_restart.npz"),
        n=N,
        kappa=KAPPA,
        ps=np.array(PS),
        replicates=REPLICATES,
    )
    print(f"\nДанные --> block_restart.npz")
    print(f"Фигура --> {OUTDIR}/fig_block_restart.pdf")

    # Сводная таблица по факторам
    print("\n=== Фактор ускорения K(Broyden) / K(method) ===")
    K_br = float(np.median(results[(0, "broyden")]))
    print(f"K(Broyden) = {K_br:.0f}")
    print(f"{'p':>4s} | {'rolling':>9s} | {'block-rest':>11s} | {'random':>8s} | {'theory':>8s}")
    print("-" * 58)
    for p in PS:
        K_roll = float(np.median(results[(p, "rolling")]))
        K_block = float(np.median(results[(p, "block_restart")]))
        K_rand = float(np.median(results[(p, "random_sketch")]))
        print(
            f"{p:>4d} | {K_br/K_roll:>8.2f}x | {K_br/K_block:>10.2f}x | "
            f"{K_br/K_rand:>7.2f}x | {(p+1):>7.2f}x"
        )


if __name__ == "__main__":
    main()
