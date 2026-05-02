"""
diag_jacerr_stat.py — статистически корректная версия fig:spb_jacerr.

Discrete BVP, n=100. Берём 20 случайных возмущений дефолтного x_0 из MGH:
  x_0 = x_0^default + 0.05 * u,  u ~ N(0, I_n)/||u||,
запускаем три метода (классический Бройден, SP-Broyden p<=5, p<=10),
записываем ||B_k - J(x_k)||_F вдоль траектории. Затем строим медианную
кривую + IQR-затенение по 20 запускам для каждого метода.

Выход: mipt_thesis_master/fig_sp_broyden_jacerr.pdf  (перезаписывает старый
двухпанельный график).
"""
from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Discrete BVP (MGH #28) ----------

def discrete_bvp_F(x):
    n = len(x); h = 1.0 / (n + 1)
    r = np.zeros(n)
    for i in range(n):
        ti = (i + 1) * h
        xm = x[i - 1] if i > 0 else 0.0
        xp = x[i + 1] if i < n - 1 else 0.0
        r[i] = 2.0 * x[i] - xm - xp + h * h * (x[i] + ti + 1.0) ** 3 / 2.0
    return r


def discrete_bvp_J(x):
    n = len(x); h = 1.0 / (n + 1)
    J = np.zeros((n, n))
    for i in range(n):
        ti = (i + 1) * h
        J[i, i] = 2.0 + h * h * 3.0 * (x[i] + ti + 1.0) ** 2 / 2.0
        if i > 0: J[i, i - 1] = -1.0
        if i < n - 1: J[i, i + 1] = -1.0
    return J


def discrete_bvp_x0_default(n):
    h = 1.0 / (n + 1)
    return 0.1 * np.array([i * h * (i * h - 1.0) for i in range(1, n + 1)])


# ---------- солвер с трекингом ||B_k - J(x_k)||_F ----------

def sp_broyden_track(F, Jf, x0, p_max=0, cond_thresh=1e3,
                      maxit=600, tol=1e-13):
    """Возвращает массив jac_err: ||B_k - J(x_k)||_F по итерациям.
    Длина массива = число выполненных итераций (или maxit при отказе).
    Останавливается при ||F||<tol.
    """
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    B = np.eye(n)
    S_hist = []
    jac_err = [norm(np.eye(n) - Jf(x), ord="fro")]
    for k in range(maxit):
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
        if not np.all(np.isfinite(Fx_new)):
            break
        s = d; y = Fx_new - Fx
        S_hist.append(s.copy())
        # adaptive p
        p_eff = 0
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                if cond(Sp.T @ Sp) < cond_thresh:
                    p_eff = p_try
                else:
                    break
        if p_eff == 0:
            v = s
        else:
            cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols); G = Sp.T @ Sp
            e1 = np.zeros(p_eff + 1); e1[0] = 1.0
            try:
                v = Sp @ solve(G, e1)
            except np.linalg.LinAlgError:
                v = s
        denom = float(v @ s)
        if abs(denom) < 1e-14:
            break
        Bs = B @ s
        B = B + np.outer(y - Bs, v) / denom
        x, Fx = x_new, Fx_new
        jac_err.append(norm(B - Jf(x), ord="fro"))
        if len(S_hist) > p_max + 5:
            S_hist.pop(0)
    return np.array(jac_err)


# ---------- эксперимент ----------

def main():
    n = 100
    n_seeds = 20
    eps = 0.05               # амплитуда возмущения x_0
    rng = np.random.default_rng(20260502)
    x0_default = discrete_bvp_x0_default(n)

    methods = [
        ("Бройден",        dict(p_max=0)),
        ("SP-Broyden, $p\\leq 5$",  dict(p_max=5)),
        ("SP-Broyden, $p\\leq 10$", dict(p_max=10)),
    ]
    style = {
        "Бройден":               dict(color="#666666", ls=(0, (5, 3)), lw=1.4),
        "SP-Broyden, $p\\leq 5$": dict(color="#2060B0", ls=(0, (4, 2)), lw=1.6),
        "SP-Broyden, $p\\leq 10$":dict(color="#D03030", ls="-",         lw=2.0),
    }

    # сгенерируем общие случайные направления для всех методов
    dirs = rng.standard_normal((n_seeds, n))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    starts = [x0_default + eps * u for u in dirs]

    runs = {name: [] for name, _ in methods}

    print(f"Запускаем {n_seeds} стартов на 3 методах...")
    for name, kw in methods:
        for x0 in starts:
            traj = sp_broyden_track(discrete_bvp_F, discrete_bvp_J,
                                     x0, maxit=600, tol=1e-13, **kw)
            runs[name].append(traj)
        lengths = [len(t) for t in runs[name]]
        print(f"  {name:<22s}  ит. до сход.: median={int(np.median(lengths))}, "
              f"min={min(lengths)}, max={max(lengths)}")

    # ---------- картинка: медиана + IQR ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, _ in methods:
        trajs = runs[name]
        L = max(len(t) for t in trajs)
        # выровняем NaN после сходимости
        M = np.full((n_seeds, L), np.nan)
        for i, t in enumerate(trajs):
            M[i, :len(t)] = t
        median = np.nanmedian(M, axis=0)
        q25 = np.nanquantile(M, 0.25, axis=0)
        q75 = np.nanquantile(M, 0.75, axis=0)
        ks = np.arange(L)
        # обрезать хвост, где осталось < 50% запусков (для аккуратности)
        valid = (~np.isnan(M)).sum(axis=0) >= n_seeds // 2
        last = np.where(valid)[0][-1] + 1 if valid.any() else 0
        sl = slice(0, last)
        ax.fill_between(ks[sl], q25[sl], q75[sl],
                        color=style[name]["color"], alpha=0.18, lw=0)
        ax.plot(ks[sl], median[sl], label=name, **style[name])

    ax.set_yscale("log")
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|B_k - J(x_k)\|_F$")
    ax.set_title(rf"Discrete BVP, $n=100$: ограниченность $\|E_k\|_F$ "
                 rf"(медиана$\pm$IQR, {n_seeds} стартов)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mipt_thesis_master", "fig_sp_broyden_jacerr.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()
