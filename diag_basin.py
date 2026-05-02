"""
diag_basin.py — построение области (бассейна) сходимости трёх методов на
Discrete BVP, n=100. Реализует тот же эксперимент-сравнение, что fig:spb_conv,
но методологически устойчивее: один прогон из одного x_0 заменяется на
статистику по случайным направлениям и сетке радиусов R.

Идея. Найдём x* высокоточным прогоном SP-Broyden. Возьмём log-сетку
радиусов R; для каждого R и каждого случайного единичного направления u
запустим метод из x_0 = x* + R u и зарегистрируем, сошёлся ли он
(||F||<tol за <=maxit, без NaN/Inf). Доля сошедшихся запусков как функция
R даёт эмпирическую "вероятность сходимости" — сравнимую между методами.

Все методы используют B_0 = I, без line search; SP-Broyden — с адаптивным
порогом cond(S_p^T S_p) < 1e3 и p_max = 5; Anderson — Walker-Ni Type-II,
m=5, beta=1.0, tau=1/||J*||, residual-safeguard.

Выход (mipt_thesis_master/):
  fig_spb_basin.pdf   — две панели: (a) доля сошедшихся запусков vs R,
                        (b) распределение #итераций до сходимости.
  basin_results.npz   — сырые таблицы (метод, R, направление, сошёлся, #it).
"""
from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- задача Discrete BVP, n=100 (Moré-Garbow-Hillstrom #28) ----------

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


def discrete_bvp_x0_default(n: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    return 0.1 * np.array([i * h * (i * h - 1.0) for i in range(1, n + 1)])


# ---------- солверы ----------
# Везде: B_0 = I, без line search, без fallback'ов; критерий
# "сошёлся": ||F||<tol за <=maxit и все траектории конечны.

STEP_CAP = 1e3   # отсекаем взрывные шаги ||d||>STEP_CAP*max(||x||,1)

def _step_safe(x, d):
    nrm = norm(d)
    cap = STEP_CAP * max(norm(x), 1.0)
    if not np.isfinite(nrm) or nrm > cap:
        return False
    return True


def broyden_solve(F, x0, maxit=300, tol=1e-8):
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return False, 0
    B = np.eye(n)
    for k in range(maxit):
        if norm(Fx) < tol:
            return True, k
        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            return False, k
        if not _step_safe(x, d):
            return False, k
        x_new = x + d
        Fx_new = F(x_new)
        if not np.all(np.isfinite(Fx_new)):
            return False, k
        s = d
        y = Fx_new - Fx
        Bs = B @ s
        denom = float(s @ s)
        if abs(denom) < 1e-14:
            return False, k
        B = B + np.outer(y - Bs, s) / denom
        x, Fx = x_new, Fx_new
    return norm(Fx) < tol, maxit


def sp_broyden_solve(F, x0, p_max=5, cond_thresh=1e3, maxit=300, tol=1e-8):
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return False, 0
    B = np.eye(n)
    S_hist = []
    for k in range(maxit):
        if norm(Fx) < tol:
            return True, k
        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            return False, k
        if not _step_safe(x, d):
            return False, k
        x_new = x + d
        Fx_new = F(x_new)
        if not np.all(np.isfinite(Fx_new)):
            return False, k
        s = d
        y = Fx_new - Fx
        S_hist.append(s.copy())
        # adaptive p
        p_eff = 0
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                if cond(G) < cond_thresh:
                    p_eff = p_try
                else:
                    break
        if p_eff == 0:
            v = s
        else:
            cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols)
            G = Sp.T @ Sp
            e1 = np.zeros(p_eff + 1); e1[0] = 1.0
            try:
                v = Sp @ solve(G, e1)
            except np.linalg.LinAlgError:
                v = s
        denom = float(v @ s)
        if abs(denom) < 1e-14:
            return False, k
        Bs = B @ s
        B = B + np.outer(y - Bs, v) / denom
        x, Fx = x_new, Fx_new
        if len(S_hist) > p_max + 5:
            S_hist.pop(0)
    return norm(Fx) < tol, maxit


def anderson_solve(F, x0, m=5, beta=1.0, tau=0.2, maxit=300, tol=1e-8,
                   eta_safe=2.0, beta_min=1e-3):
    """Anderson Type-II с residual-safeguard (Walker–Ni 2011)."""
    x = x0.astype(float).copy()
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return False, 0
    f_curr = -tau * Fx
    X_hist = [x.copy()]
    F_hist = [f_curr.copy()]
    last_res = norm(Fx)
    for k in range(maxit):
        if norm(Fx) < tol:
            return True, k
        m_k = min(m, len(X_hist) - 1)

        def step(beta_local):
            if m_k == 0:
                return x + beta_local * f_curr
            X_arr = np.column_stack(X_hist[-(m_k + 1):])
            F_arr = np.column_stack(F_hist[-(m_k + 1):])
            DX = X_arr[:, 1:] - X_arr[:, :-1]
            DF = F_arr[:, 1:] - F_arr[:, :-1]
            G = DF.T @ DF
            G = G + 1e-10 * (np.trace(G) + 1.0) * np.eye(m_k)
            try:
                gamma = solve(G, DF.T @ f_curr)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_k)
            return x + beta_local * f_curr - (DX + beta_local * DF) @ gamma

        b = beta
        x_new = step(b)
        Fx_new = F(x_new) if np.all(np.isfinite(x_new)) else None
        while (Fx_new is None or not np.all(np.isfinite(Fx_new))
               or norm(Fx_new) > eta_safe * last_res):
            b *= 0.5
            if b < beta_min:
                # safe Picard restart
                X_hist = [x.copy()]; F_hist = [f_curr.copy()]
                x_new = x - tau * Fx
                Fx_new = F(x_new)
                break
            x_new = step(b)
            Fx_new = F(x_new) if np.all(np.isfinite(x_new)) else None
        if Fx_new is None or not np.all(np.isfinite(Fx_new)):
            return False, k
        if not _step_safe(x, x_new - x):
            return False, k
        f_new = -tau * Fx_new
        X_hist.append(x_new.copy()); F_hist.append(f_new.copy())
        if len(X_hist) > m + 1:
            X_hist.pop(0); F_hist.pop(0)
        x = x_new; f_curr = f_new; Fx = Fx_new
        last_res = norm(Fx)
    return norm(Fx) < tol, maxit


# ---------- эксперимент ----------

def find_x_star(n=100):
    """Долгий запуск SP-Broyden из стандартного x0 — корень с двойной точностью."""
    F = discrete_bvp_F
    x0 = discrete_bvp_x0_default(n)
    # солвер с большим maxit и жёстким tol
    x = x0.astype(float).copy()
    Fx = F(x)
    B = np.eye(n)
    S_hist = []
    p_max, cond_thresh = 5, 1e3
    for _ in range(2000):
        if norm(Fx) < 1e-13:
            break
        d = solve(B, -Fx)
        x_new = x + d
        Fx_new = F(x_new)
        s = d; y = Fx_new - Fx
        S_hist.append(s.copy())
        p_eff = 0
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols); G = Sp.T @ Sp
                if cond(G) < cond_thresh:
                    p_eff = p_try
                else:
                    break
        if p_eff == 0:
            v = s
        else:
            cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols); G = Sp.T @ Sp
            e1 = np.zeros(p_eff + 1); e1[0] = 1.0
            v = Sp @ solve(G, e1)
        denom = float(v @ s)
        Bs = B @ s
        B = B + np.outer(y - Bs, v) / denom
        x, Fx = x_new, Fx_new
        if len(S_hist) > p_max + 5:
            S_hist.pop(0)
    return x


def main():
    n = 100
    print("Поиск x* для Discrete BVP, n=100 ...")
    x_star = find_x_star(n)
    print(f"  ||F(x*)|| = {norm(discrete_bvp_F(x_star)):.3e}")
    print(f"  ||x*||    = {norm(x_star):.4f}")

    # log-сетка радиусов
    R_grid = np.logspace(-2, 2, 17)   # [0.01, 100], 17 точек
    n_dirs = 25                       # случайных направлений на каждом R
    rng = np.random.default_rng(20260502)

    methods = [
        ("Бройден",        broyden_solve,  {}),
        ("SP-Broyden",     sp_broyden_solve, dict(p_max=5)),
        ("Anderson",       anderson_solve,   dict(m=5, beta=1.0, tau=0.2)),
    ]

    # фиксируем направления (одни и те же для всех методов и всех R)
    dirs = rng.standard_normal((n_dirs, n))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # таблица результатов: (метод, R, направление) -> (сошёлся, #it)
    results = {name: {"converged": np.zeros((len(R_grid), n_dirs), dtype=bool),
                      "iters":     np.zeros((len(R_grid), n_dirs), dtype=int)}
               for name, _, _ in methods}

    for ir, R in enumerate(R_grid):
        for id_, u in enumerate(dirs):
            x0 = x_star + R * u
            for name, solve_fn, kw in methods:
                ok, it = solve_fn(discrete_bvp_F, x0, maxit=300, tol=1e-8, **kw)
                results[name]["converged"][ir, id_] = ok
                results[name]["iters"][ir, id_] = it
        print(f"  R = {R:>8.4f}: " + ", ".join(
            f"{name} {results[name]['converged'][ir].mean()*100:>3.0f}%"
            for name, _, _ in methods))

    # сохраняем сырые
    out_npz = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "basin_results.npz")
    np.savez_compressed(out_npz, R_grid=R_grid,
                        **{f"{name}_converged": results[name]["converged"]
                           for name, _, _ in methods},
                        **{f"{name}_iters": results[name]["iters"]
                           for name, _, _ in methods})
    print(f"saved: {out_npz}")

    # ---------- картинка ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    style = {
        "Бройден":     dict(color="#888888", ls=(0, (4, 3)), lw=1.6, marker="s", ms=4),
        "Anderson":    dict(color="#2060B0", ls=(0, (5, 2)), lw=1.6, marker="^", ms=4),
        "SP-Broyden":  dict(color="#D03030", ls="-",        lw=2.0, marker="o", ms=4),
    }
    ax = axes[0]
    for name, _, _ in methods:
        frac = results[name]["converged"].mean(axis=1)
        ax.semilogx(R_grid, frac * 100, label=name, **style[name])
    ax.set_xlabel(r"радиус $R = \|x_0 - x^{*}\|$")
    ax.set_ylabel(r"доля сошедшихся запусков, \%")
    ax.set_ylim(-3, 105)
    ax.set_title(f"Discrete BVP, $n=100$: бассейн сходимости ({n_dirs} направлений)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[1]
    for name, _, _ in methods:
        med = np.where(results[name]["converged"], results[name]["iters"], np.nan)
        med = np.nanmedian(med, axis=1)
        ax.semilogx(R_grid, med, label=name, **style[name])
    ax.set_xlabel(r"радиус $R = \|x_0 - x^{*}\|$")
    ax.set_ylabel(r"медиана числа итераций до $\|F\|<10^{-8}$")
    ax.set_title("стоимость сходимости внутри бассейна")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mipt_thesis_master", "fig_spb_basin.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_pdf}")

    # короткое резюме
    print("\nРезюме (медианный % сходимости по всем R):")
    for name, _, _ in methods:
        print(f"  {name:<14s}  R_50% = {_r_at_threshold(R_grid, results[name]['converged']):>7.2f}")


def _r_at_threshold(R_grid, conv_table, thr=0.5):
    """Возвращает наибольший R, при котором ещё >= thr запусков сходятся."""
    frac = conv_table.mean(axis=1)
    idx = np.where(frac >= thr)[0]
    if len(idx) == 0:
        return float("nan")
    return float(R_grid[idx[-1]])


if __name__ == "__main__":
    main()
