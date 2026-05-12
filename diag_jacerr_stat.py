"""
diag_jacerr_stat.py — статистически корректная версия fig:spb_jacerr.

Discrete BVP, n=100. Берём 20 случайных возмущений дефолтного x_0 из MGH:
  x_0 = x_0^default + 0.05 * u,  u ~ N(0, I_n)/||u||,
запускаем три метода (классический Бройден, Projected Broyden p<=5, p<=10),
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

def sp_broyden_track(F, Jf, x0, p_max=0, maxit=600, tol=1e-10):
    """Возвращает массив jac_err: ||B_k - J(x_k)||_F по итерациям.
    Длина массива = число выполненных итераций (или maxit при отказе).
    Возвращает также флаг сходимости (||F||<tol) для статистики.

    На траекториях, где B численно расходится (классический Бройден,
    p_max=0), промежуточные операции matmul/solve производят overflow:
    результат корректно обрабатывается проверкой `np.isfinite` ниже,
    но без `errstate` numpy сыпет RuntimeWarning'ами. Глушим их
    локально на блоке обновления."""
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    B = np.eye(n)
    S_hist = []
    # относительная ошибка якобиана: ||B_k - J(x_k)||_F / ||J(x_k)||_F
    Jx0 = Jf(x)
    jac_err = [norm(np.eye(n) - Jx0, ord="fro") / norm(Jx0, ord="fro")]
    # относительная невязка: ||F(x_k)|| / ||F(x_0)||
    F0_norm = float(norm(Fx))
    res_norm = [1.0]
    converged = False
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        for k in range(maxit):
            if norm(Fx) < tol:
                converged = True
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
            # p = min(p_max, k) — без адаптивного отбора
            p_eff = min(p_max, len(S_hist) - 1)
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
            Jx = Jf(x)
            jac_err.append(norm(B - Jx, ord="fro") / norm(Jx, ord="fro"))
            res_norm.append(float(norm(Fx)) / F0_norm)
            if len(S_hist) > p_max + 5:
                S_hist.pop(0)
    return np.array(jac_err), np.array(res_norm), F0_norm, converged


# ---------- эксперимент ----------

def main():
    n = 100
    n_seeds = 20
    eps = 0.05               # амплитуда возмущения x_0
    rng = np.random.default_rng(20260502)
    x0_default = discrete_bvp_x0_default(n)

    # сетка по p_max: видно trade-off
    p_values = [0, 1, 2, 5, 10]
    cmap = plt.get_cmap("viridis")
    methods = []
    for i, p in enumerate(p_values):
        name = (r"Бройден ($p_{\max}=0$)" if p == 0
                else rf"PB, $p_{{\max}}={p}$")
        methods.append((name, dict(p_max=p)))
    style = {}
    for i, (name, _) in enumerate(methods):
        color = cmap(i / max(1, len(p_values) - 1))
        ls = (0, (5, 3)) if i == 0 else "-"
        lw = 1.4 if i == 0 else 1.6
        style[name] = dict(color=color, ls=ls, lw=lw)

    # сгенерируем общие случайные направления для всех методов
    dirs = rng.standard_normal((n_seeds, n))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    starts = [x0_default + eps * u for u in dirs]

    runs = {name: [] for name, _ in methods}      # rel ||E||_F / ||J||_F
    runs_F = {name: [] for name, _ in methods}    # rel ||F|| / ||F_0||
    conv_stats = {name: [] for name, _ in methods}
    F0_norms = []

    print(f"Запускаем {n_seeds} стартов на {len(methods)} методах "
          f"(tol=1e-10, maxit=600)...")
    for name, kw in methods:
        for x0 in starts:
            traj, traj_F, F0_n, conv = sp_broyden_track(
                discrete_bvp_F, discrete_bvp_J,
                x0, maxit=600, tol=1e-10, **kw)
            runs[name].append(traj)
            runs_F[name].append(traj_F)
            conv_stats[name].append(conv)
            F0_norms.append(F0_n)
        lengths = [len(t) for t in runs[name]]
        n_conv = sum(conv_stats[name])
        print(f"  {name:<25s}  ит. до выхода: median={int(np.median(lengths))}, "
              f"min={min(lengths)}, max={max(lengths)}, "
              f"сошлись: {n_conv}/{n_seeds}")
    F0_median = float(np.median(F0_norms))
    print(f"  median ||F(x_0)||_2 = {F0_median:.3e}")

    # ---------- картинка: 1×2 (||E||_F и ||F||_2) ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    def plot_panel(ax, runs_dict, ylabel, title):
        for name, _ in methods:
            trajs = runs_dict[name]
            L = max(len(t) for t in trajs)
            M = np.full((n_seeds, L), np.nan)
            for i, t in enumerate(trajs):
                M[i, :len(t)] = t
            median = np.nanmedian(M, axis=0)
            q25 = np.nanquantile(M, 0.25, axis=0)
            q75 = np.nanquantile(M, 0.75, axis=0)
            ks = np.arange(L)
            valid = (~np.isnan(M)).sum(axis=0) >= n_seeds // 2
            last = np.where(valid)[0][-1] + 1 if valid.any() else 0
            sl = slice(0, last)
            ax.fill_between(ks[sl], q25[sl], q75[sl],
                            color=style[name]["color"], alpha=0.18, lw=0)
            ax.plot(ks[sl], median[sl], label=name, **style[name])
        ax.set_yscale("log")
        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)

    # (а) относительная ошибка якобиана: ||B-J||_F / ||J||_F
    plot_panel(axes[0],
                runs,
                r"$\|B_k - J(x_k)\|_F \,/\, \|J(x_k)\|_F$",
                r"(а) относительная ошибка аппроксимации якобиана")
    axes[0].legend(fontsize=8.5, loc="upper right", ncol=2)

    # (б) относительная невязка: ||F(x_k)|| / ||F(x_0)||
    plot_panel(axes[1],
                runs_F,
                r"$\|F(x_k)\|_2 \,/\, \|F(x_0)\|_2$",
                r"(б) относительная невязка")
    # tol на уровне абсолютной ||F||<1e-10, относительной = 1e-10/median(||F_0||)
    rel_tol = 1e-10 / F0_median
    axes[1].axhline(rel_tol, color="#888", lw=0.7, ls=":")
    axes[1].text(axes[1].get_xlim()[1] * 0.98, rel_tol * 1.5,
                  fr"tol $=10^{{-10}} / \|F(x_0)\|_2 \approx {rel_tol:.0e}$",
                  ha="right", va="bottom", fontsize=8, color="#666")

    fig.suptitle(rf"Discrete BVP, $n=100$, "
                  rf"$\mathtt{{maxit}}=600$, медиана$\pm$IQR по {n_seeds} стартам",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.tight_layout()

    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mipt_thesis_master", "fig_sp_broyden_jacerr.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()
