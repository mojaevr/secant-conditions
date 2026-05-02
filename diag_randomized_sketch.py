"""
Численная проверка (p+1)x теоремы для РАНДОМИЗИРОВАННОГО SP-Broyden.

Алгоритм (sketch-and-project, Гауэр-Рихтарик 2015):
  на шаге k берём S_k = (v_1, ..., v_{p+1}) ~ N(0, I)^{n x (p+1)},
  считаем Y_k = A S_k (один матвекс на колонку),
  обновляем B_{k+1} = arg min ||B - B_k||_F  s.t.  B S_k = Y_k.

Это RANDOMIZED rank-(p+1) sketch-and-project. В отличие от траекторного
rolling-window SP-Broyden, тут проекторы Pi_k = S_k(S_k^T S_k)^{-1}S_k^T
независимы, и теорема Гауэра-Рихтарика даёт ТОЧНОЕ:

    E[||B_{k+1} - A||_F^2] = (1 - (p+1)/n) * E[||B_k - A||_F^2].

Цель: проверить, что эмпирическая скорость убывания
||B_k - A||_F^2 ложится на теоретическую прямую с наклоном
log(1 - (p+1)/n) в полу-логарифмическом масштабе.

Этот эксперимент проверяет (p+1)x фактор ИМЕННО на той версии
алгоритма, для которой он доказывается. Положительный результат
показывает, что теория корректна; ограничение детерминированной
версии --- структурное (rolling window не даёт независимых проекторов).

Output:
  fig_random_sketch_decay.pdf   --- ||B_k - A||_F vs k для разных p
  fig_random_sketch_rate.pdf    --- эмпирический наклон vs теоретический
  random_sketch.npz             --- сырые данные
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import norm, solve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_well_conditioned_A(n, kappa, rng):
    """A = Q1 diag(geomspace(1, 1/kappa, n)) Q2^T --- нс невырожденная."""
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sigmas = np.geomspace(1.0, 1.0 / kappa, n)
    return Q1 @ np.diag(sigmas) @ Q2.T


def randomized_sketch_update(A, B0, p, n_steps, rng):
    """Чистая sketch-and-project итерация (без Newton-step, только для B).

    На каждом шаге: S_k случайный гауссовский (n x (p+1)), Y_k = A S_k,
    B_{k+1} = B_k + (Y_k - B_k S_k) (S_k^T S_k)^{-1} S_k^T.

    Возвращает массив ||B_k - A||_F^2 длины n_steps+1.
    """
    n = A.shape[0]
    B = B0.copy()
    err_sq = [float(norm(B - A, "fro") ** 2)]

    for k in range(n_steps):
        S = rng.standard_normal((n, p + 1))
        Y = A @ S
        BS = B @ S
        G = S.T @ S
        try:
            # B + (Y - BS) G^{-1} S^T
            update = (Y - BS) @ solve(G, S.T)
            B = B + update
        except np.linalg.LinAlgError:
            break
        err_sq.append(float(norm(B - A, "fro") ** 2))

    return np.array(err_sq)


def main():
    OUTDIR = os.path.join(os.path.dirname(__file__), "mipt_thesis_master")
    SEED = 20260502
    NS = [50, 100, 200]
    PS = [0, 1, 2, 5, 10, 20]
    KAPPA = 5.0
    REPLICATES = 30  # для усреднения по случайным S_k
    N_STEPS_FN = lambda n: 4 * n  # достаточно для (1-(p+1)/n)^k -> 1e-12

    decay_data = {}  # (n, p) -> array of shape (n_steps+1,) average err^2
    decay_traces = {}  # (n, p) -> single trace for visualization

    print("Запуск рандомизированного SP-Broyden эксперимента...")
    for n in NS:
        n_steps = N_STEPS_FN(n)
        for p in PS:
            traces = []
            for r in range(REPLICATES):
                rng = np.random.default_rng(
                    SEED + 1000 * NS.index(n) + 17 * p + r
                )
                A = make_well_conditioned_A(n, KAPPA, rng)
                B0 = np.eye(n)
                err_sq = randomized_sketch_update(A, B0, p, n_steps, rng)
                traces.append(err_sq)
            # усредняем
            avg = np.mean(traces, axis=0)
            decay_data[(n, p)] = avg
            decay_traces[(n, p)] = traces[0]
            # эмпирический наклон в полу-лог: лин. регрессия log(err) vs k
            half = max(2, n_steps // 2)
            log_avg = np.log(np.maximum(avg[:half + 1], 1e-300))
            ks = np.arange(half + 1)
            valid = (avg[:half + 1] > 1e-12)
            if valid.sum() > 5:
                slope, _ = np.polyfit(ks[valid], log_avg[valid], 1)
            else:
                slope = float("nan")
            theory_slope = np.log(1 - (p + 1) / n)
            ratio = slope / theory_slope if theory_slope < 0 else float("nan")
            print(
                f"  n={n:3d}, p={p:2d}: empirical slope={slope:.5f}, "
                f"theory log(1-(p+1)/n)={theory_slope:.5f}, ratio={ratio:.3f}"
            )

    # ============ Фигура 1: убывание ||B_k - A||_F^2 ============
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(PS)))
    for ax, n in zip(axes, NS):
        n_steps = N_STEPS_FN(n)
        for c, p in zip(cmap, PS):
            avg = decay_data[(n, p)]
            ks = np.arange(len(avg))
            ax.semilogy(ks, np.maximum(avg, 1e-15), color=c, label=f"$p={p}$")
            # теоретическая прямая: avg[0] * (1 - (p+1)/n)^k
            theory = avg[0] * (1 - (p + 1) / n) ** ks
            ax.semilogy(ks, np.maximum(theory, 1e-15), color=c, linestyle="--",
                        alpha=0.5, linewidth=0.8)
        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(r"$\mathbb{E}[\|B_k - A\|_F^2]$ (среднее по %d запускам)" % REPLICATES)
        ax.set_title(f"$n={n}$, $\kappa(A)={KAPPA}$")
        ax.set_ylim(1e-12, None)
        ax.set_xlim(0, n_steps)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    # пунктир в легенде
    axes[0].plot([], [], color="gray", linestyle="--",
                 label=r"теория $(1-\frac{p+1}{n})^k$")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_random_sketch_decay.pdf"))
    plt.close(fig)

    # ============ Фигура 2: эмпирический наклон vs теоретический ============
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for n in NS:
        emp_slopes = []
        theory_slopes = []
        for p in PS:
            avg = decay_data[(n, p)]
            half = max(2, len(avg) // 2)
            log_avg = np.log(np.maximum(avg[:half + 1], 1e-300))
            ks = np.arange(half + 1)
            valid = (avg[:half + 1] > 1e-12)
            if valid.sum() > 5:
                slope, _ = np.polyfit(ks[valid], log_avg[valid], 1)
                emp_slopes.append(-slope)
            else:
                emp_slopes.append(float("nan"))
            theory_slopes.append(-np.log(1 - (p + 1) / n))
        ax.plot(theory_slopes, emp_slopes, "o-", label=f"$n={n}$")
    diag = np.linspace(0, max(theory_slopes) * 1.1, 50)
    ax.plot(diag, diag, "k--", alpha=0.5, label="diagonal $y=x$")
    ax.set_xlabel(r"теоретический наклон $-\log(1-(p+1)/n)$")
    ax.set_ylabel(r"эмпирический наклон")
    ax.set_title("Sketch-and-project: эмпирическая скорость убывания\n"
                 r"совпадает с теоретическим $(1-(p+1)/n)^k$ фактором")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_random_sketch_rate.pdf"))
    plt.close(fig)

    # сохраняем
    np.savez(
        os.path.join(os.path.dirname(__file__), "random_sketch.npz"),
        ns=np.array(NS),
        ps=np.array(PS),
        kappa=KAPPA,
        replicates=REPLICATES,
    )
    print(f"\nДанные --> random_sketch.npz")
    print(f"Фигуры --> {OUTDIR}/fig_random_sketch_{{decay,rate}}.pdf")


if __name__ == "__main__":
    main()
