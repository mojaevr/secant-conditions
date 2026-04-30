"""
Диагностические эксперименты SP-Broyden для главы 2 диссертации.
Закрывает три пункта P1 (численные эксперименты, гл. 2):
  1) панель ||B_k - J(x_k)||_F по итерациям (проверка bounded deterioration);
  2) сходимость при p in {1, 2, 5, 10, 20} (зависимость от глубины окна);
  3) cond(S_p^T S_p) по итерациям (как часто адаптивный порог сужает окно).

Стандартные тестовые задачи Море-Гарбов-Хилстрём (J. Moré et al., 1981, n=100):
Discrete BVP и Broyden Banded. Используется тот же солвер, что в tezisy.ipynb,
расширенный записью B_k, S_p, выбранного p и cond(S_p^T S_p) на каждой итерации.

Выход (mipt_thesis_master/):
  fig_sp_broyden_jacerr.pdf   — ||B_k - J(x_k)||_F и ||E_k Pi_k||_F
  fig_sp_broyden_pvar.pdf     — ||F(x_k)|| для p in {1,2,5,10,20}
  fig_sp_broyden_cond.pdf     — cond(S_p^T S_p), эффективное p, частота сужения

Воспроизводимость: NumPy 1.26+, фиксированный seed для всех источников
случайности (здесь стартовая точка детерминирована, но np.random.default_rng
оставлен для совместимости с reproducibility.tex).
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Тестовые задачи: F(x) и аналитический якобиан J(x).
# ----------------------------------------------------------------------


def discrete_bvp_F(x: np.ndarray) -> np.ndarray:
    n = len(x)
    h = 1.0 / (n + 1)
    r = np.zeros(n)
    for i in range(n):
        ti = (i + 1) * h
        xm = x[i - 1] if i > 0 else 0.0
        xp = x[i + 1] if i < n - 1 else 0.0
        r[i] = 2.0 * x[i] - xm - xp + h * h * (x[i] + ti + 1.0) ** 3 / 2.0
    return r


def discrete_bvp_J(x: np.ndarray) -> np.ndarray:
    n = len(x)
    h = 1.0 / (n + 1)
    J = np.zeros((n, n))
    for i in range(n):
        ti = (i + 1) * h
        J[i, i] = 2.0 + h * h * 3.0 * (x[i] + ti + 1.0) ** 2 / 2.0
        if i > 0:
            J[i, i - 1] = -1.0
        if i < n - 1:
            J[i, i + 1] = -1.0
    return J


def discrete_bvp_x0(n: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    return 0.1 * np.array([i * h * (i * h - 1.0) for i in range(1, n + 1)])


def broyden_banded_F(x: np.ndarray) -> np.ndarray:
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        ji = [j for j in range(max(0, i - 5), min(n, i + 2)) if j != i]
        r[i] = x[i] * (2.0 + 5.0 * x[i] ** 2) + 1.0 - sum(
            x[j] * (1.0 + x[j]) for j in ji
        )
    return r


def broyden_banded_J(x: np.ndarray) -> np.ndarray:
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2.0 + 15.0 * x[i] ** 2
        for j in range(max(0, i - 5), min(n, i + 2)):
            if j == i:
                continue
            J[i, j] = -(1.0 + 2.0 * x[j])
    return J


def broyden_banded_x0(n: int) -> np.ndarray:
    return -0.1 * np.ones(n)


# ----------------------------------------------------------------------
# SP-Broyden / классический Бройден с трекингом диагностических величин.
# ----------------------------------------------------------------------


def sp_broyden_solve(
    F,
    x0: np.ndarray,
    p_max: int = 0,
    cond_thresh: float = 1e3,
    maxiter: int = 600,
    tol: float = 1e-13,
):
    """Возвращает словарь со списками по итерациям:
    iters, fevals, res (||F||), p_eff, cond_Sp, jac_err (||B_k - J(x_k)||_F),
    Eproj (||E_k Pi_k||_F).  Якобиан J считается по аналитической формуле,
    переданной в `Jfun` через замыкание (см. ниже).
    """
    raise NotImplementedError


# Чтобы избежать таскания Jfun через много слоёв, делаем функтор:
def make_solver(Jfun):
    def solve_with_diag(
        F,
        x0,
        p_max=0,
        cond_thresh=1e3,
        maxiter=600,
        tol=1e-13,
        record_jac_err=False,
    ):
        n = len(x0)
        x = x0.copy().astype(float)
        Fx = F(x)
        f_evals = 1
        out = {
            "iters": [0],
            "fevals": [1],
            "res": [float(norm(Fx))],
            "p_eff": [0],
            "cond_Sp": [0.0],
            "jac_err": [],
            "Eproj": [],
        }
        if record_jac_err:
            J0 = Jfun(x)
            out["jac_err"].append(float(norm(np.eye(n) - J0, ord="fro")))
            out["Eproj"].append(0.0)

        B = np.eye(n)
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
            Fx_new = F(x_new)
            f_evals += 1
            if not np.all(np.isfinite(Fx_new)):
                break
            y = Fx_new - Fx
            s = d
            S_hist.append(s.copy())
            Y_hist.append(y.copy())

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
            if abs(denom) < 1e-12:
                v = s
                denom = float(s @ s)

            # ---- запись Pi_k и ||E_k Pi_k||_F (до обновления B!) ----
            if record_jac_err:
                Jx = Jfun(x)
                E = B - Jx
                if p_eff == 0:
                    # Pi_k = s s^T / (s^T s)
                    Es = E @ s
                    Eproj = float(norm(Es) / np.sqrt(s @ s))
                else:
                    cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
                    Sp = np.column_stack(cols)
                    # Pi = Sp (Sp^T Sp)^{-1} Sp^T
                    try:
                        Q, _ = np.linalg.qr(Sp)
                        EPi = E @ Q
                        Eproj = float(norm(EPi, ord="fro"))
                    except np.linalg.LinAlgError:
                        Eproj = float("nan")
                out["Eproj"].append(Eproj)

            B = B + np.outer(y - Bs, v) / denom

            x = x_new
            Fx = Fx_new
            out["iters"].append(k + 1)
            out["fevals"].append(f_evals)
            out["res"].append(float(norm(Fx)))
            out["p_eff"].append(int(p_eff))
            out["cond_Sp"].append(cond_used)

            if record_jac_err:
                Jx_new = Jfun(x)
                out["jac_err"].append(float(norm(B - Jx_new, ord="fro")))

            max_hist = max(p_max + 5, 25)
            if len(S_hist) > max_hist:
                S_hist.pop(0)
                Y_hist.pop(0)

        return out

    return solve_with_diag


# ----------------------------------------------------------------------
# Эксперимент 1: ||B_k - J(x_k)||_F vs iterations  (Discrete BVP).
# ----------------------------------------------------------------------


def panel_jacerr():
    n = 100
    F = discrete_bvp_F
    Jf = discrete_bvp_J
    x0 = discrete_bvp_x0(n)
    solver = make_solver(Jf)

    runs = {
        "Бройден": solver(
            F, x0, p_max=0, maxiter=600, tol=1e-13, record_jac_err=True
        ),
        "SP-Broyden, p<=5": solver(
            F, x0, p_max=5, maxiter=600, tol=1e-13, record_jac_err=True
        ),
        "SP-Broyden, p<=10": solver(
            F, x0, p_max=10, maxiter=600, tol=1e-13, record_jac_err=True
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

    style = {
        "Бройден": dict(color="#888888", ls=(0, (4, 3)), lw=1.4),
        "SP-Broyden, p<=5": dict(color="#2060B0", ls=(0, (5, 2)), lw=1.6),
        "SP-Broyden, p<=10": dict(color="#D03030", ls="-", lw=2.2),
    }

    ax = axes[0]
    for label, h in runs.items():
        ax.semilogy(h["iters"], h["jac_err"], label=label, **style[label])
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|B_k - J(x_k)\|_F$")
    ax.set_title("Bounded deterioration")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for label, h in runs.items():
        if len(h["Eproj"]) >= 2:
            # Eproj[0] — фиктивное; пропустим его.
            iters = h["iters"][1:]
            vals = h["Eproj"][1:]
            ax.semilogy(iters, vals, label=label, **style[label])
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|E_k\Pi_k\|_F$")
    ax.set_title(r"снимаемая ошибка проектором $\Pi_k$")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8.5)

    fig.suptitle(
        "Discrete BVP, $n=100$: эволюция аппроксимации якобиана",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = "mipt_thesis_master/fig_sp_broyden_jacerr.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")

    print("    final ||B_k-J(x_k)||_F:")
    for label, h in runs.items():
        print(
            f"      {label:<20s}  iters={h['iters'][-1]:>4d}  "
            f"||F||={h['res'][-1]:.2e}  ||B-J||_F={h['jac_err'][-1]:.3e}"
        )


# ----------------------------------------------------------------------
# Эксперимент 2: сходимость для p in {1,2,5,10,20}.
# ----------------------------------------------------------------------


def panel_pvar():
    n = 100
    P_VALUES = [1, 2, 5, 10, 20]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(P_VALUES) - 1)) for i in range(len(P_VALUES))]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

    for ax_idx, (prob_name, F, Jf, x0, tol_target, maxit) in enumerate(
        [
            (
                "Discrete BVP, $n=100$",
                discrete_bvp_F,
                discrete_bvp_J,
                discrete_bvp_x0(n),
                1e-13,
                500,
            ),
            (
                "Broyden Banded, $n=100$",
                broyden_banded_F,
                broyden_banded_J,
                broyden_banded_x0(n),
                1e-12,
                300,
            ),
        ]
    ):
        ax = axes[ax_idx]
        solver = make_solver(Jf)
        # Бройден (p=0) для baseline:
        h0 = solver(F, x0, p_max=0, maxiter=maxit, tol=tol_target)
        ax.semilogy(
            h0["iters"], h0["res"],
            color="#888888", ls=(0, (4, 3)), lw=1.3,
            label="Бройден ($p=0$)",
        )
        for p_max, color in zip(P_VALUES, colors):
            h = solver(F, x0, p_max=p_max, maxiter=maxit, tol=tol_target)
            ax.semilogy(
                h["iters"], h["res"],
                color=color, lw=1.7, label=f"$p\\leq{p_max}$",
            )
            print(
                f"  [{prob_name}] p<={p_max:>2d}: iters={h['iters'][-1]:>4d}  "
                f"fevals={h['fevals'][-1]:>4d}  ||F||={h['res'][-1]:.2e}"
            )
        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(r"$\|F(x_k)\|$")
        ax.set_title(prob_name)
        ax.axhline(1e-10, color="k", ls=":", lw=0.7, alpha=0.6)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=8.5, ncol=2)

    fig.suptitle(
        "SP-Broyden: зависимость сходимости от глубины окна $p$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = "mipt_thesis_master/fig_sp_broyden_pvar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ----------------------------------------------------------------------
# Эксперимент 3: cond(S_p^T S_p) по итерациям + эффективное p.
# ----------------------------------------------------------------------


def panel_cond():
    n = 100
    F = discrete_bvp_F
    Jf = discrete_bvp_J
    x0 = discrete_bvp_x0(n)
    solver = make_solver(Jf)
    h = solver(F, x0, p_max=10, maxiter=400, tol=1e-13)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

    ax = axes[0]
    iters = np.array(h["iters"])
    cond_arr = np.array(h["cond_Sp"])
    mask = cond_arr > 0
    ax.semilogy(
        iters[mask], cond_arr[mask], color="#D03030", lw=1.6,
        label=r"$\mathrm{cond}(S_p^\top S_p)$",
    )
    ax.axhline(
        1e3, color="k", ls=(0, (3, 3)), lw=1.0,
        label=r"порог $\tau=10^3$",
    )
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\mathrm{cond}(S_p^\top S_p)$")
    ax.set_title("обусловленность грамиана секущих")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    p_arr = np.array(h["p_eff"])
    ax.step(
        iters, p_arr, where="post", color="#2060B0", lw=1.6,
        label=r"эффективное $p_k$",
    )
    ax.axhline(
        10, color="k", ls=(0, (3, 3)), lw=1.0,
        label=r"$p_{\max}=10$",
    )
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$p_k$")
    ax.set_title("адаптивный выбор глубины окна")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8.5)

    fig.suptitle(
        "Discrete BVP, $n=100$: контроль адаптивного порога окна",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = "mipt_thesis_master/fig_sp_broyden_cond.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")

    # короткая статистика:
    n_total = max(1, len(p_arr) - 1)  # без нулевой итерации
    n_full = int(np.sum(p_arr[1:] >= 10))
    n_shrunk = int(np.sum((p_arr[1:] > 0) & (p_arr[1:] < 10)))
    n_zero = int(np.sum(p_arr[1:] == 0))
    print(
        f"    p_eff distribution (over k=1..{n_total}): "
        f"=10 in {n_full} iters, in [1,9] in {n_shrunk}, "
        f"=0 (warmup) in {n_zero}"
    )
    print(
        f"    cond_Sp summary:  median={np.median(cond_arr[mask]):.2e}  "
        f"max={cond_arr[mask].max():.2e}"
    )


# ----------------------------------------------------------------------
# Точка входа.
# ----------------------------------------------------------------------


def main():
    np.random.seed(0)
    rng = np.random.default_rng(2026)  # для воспроизводимости-стиля
    _ = rng  # not used, but matches reproducibility.tex policy

    print("=" * 70)
    print("SP-Broyden диагностика | гл. 2 | три панели")
    print("=" * 70)
    print("[1/3] panel_jacerr (||B_k - J(x_k)||_F)")
    panel_jacerr()
    print("[2/3] panel_pvar (p in {1,2,5,10,20})")
    panel_pvar()
    print("[3/3] panel_cond (cond(S_p^T S_p))")
    panel_cond()
    print("done.")


if __name__ == "__main__":
    main()
