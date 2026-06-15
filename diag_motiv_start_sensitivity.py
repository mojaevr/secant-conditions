"""
diag_motiv_start_sensitivity.py — иллюстрация к МОТИВАЦИИ (слайд 4b).

Цель: численно показать классический результат, на который ссылается слайд 4
(Broyden–Dennis–Moré 1973; Griewank 1986): классический квазиньютоновский
(Бройден) метод сходится ЛИШЬ ЛОКАЛЬНО и чувствителен к старту — без
глобализации сходимость не гарантирована.

Стандартная тест-задача: Broyden Tridiagonal (Moré–Garbow–Hillstrom 1981, #30),
n=100. Берём корень x* (Ньютон от штатного старта MGH), затем запускаем
КЛАССИЧЕСКИЙ Бройден из стартов на разном относительном расстоянии d=||x0-x*||/||x*||
от корня (несколько случайных направлений на каждое d) и пишем ||F(x_k)||/||F(x_0)||.

Близкие старты → сверхлинейная сходимость; далёкие → застой/расходимость.

Выход: Презентация_магистр__диссертации/figs/fig_motiv_start_sensitivity.pdf
"""
from __future__ import annotations

import os
import numpy as np
from numpy.linalg import norm, solve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Broyden Tridiagonal (MGH #30) ----------

def btri_F(x):
    n = len(x)
    f = np.empty(n)
    for i in range(n):
        xm = x[i - 1] if i > 0 else 0.0
        xp = x[i + 1] if i < n - 1 else 0.0
        f[i] = (3.0 - 2.0 * x[i]) * x[i] - xm - 2.0 * xp + 1.0
    return f


def btri_J(x):
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 3.0 - 4.0 * x[i]
        if i > 0: J[i, i - 1] = -1.0
        if i < n - 1: J[i, i + 1] = -2.0
    return J


def newton(F, Jf, x0, maxit=200, tol=1e-14):
    x = x0.astype(float).copy()
    for _ in range(maxit):
        Fx = F(x)
        if norm(Fx) < tol:
            break
        x = x + solve(Jf(x), -Fx)
    return x


# ---------- классический Бройден (good), B_0 на выбор ----------

def broyden_classic(F, Jf, x0, b0_mode="J", maxit=300, tol=1e-10):
    """Классический Бройден. Возвращает (res_norm, status), где
    res_norm = ||F(x_k)||/||F(x_0)|| по итерациям, а status — одно из
    'conv' (достигнут tol), 'diverge' (overflow/NaN), 'stall' (шаг
    схлопнулся без сходимости / вырожденная B), 'maxit' (исчерпан лимит).

    Проверка сходимости стоит СРАЗУ после шага: при сверхлинейной сходимости
    последний шаг крошечный, и иначе сработал бы guard denom<1e-14 и пометил
    сошедшуюся траекторию как незавершённую."""
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    B = Jf(x).copy() if b0_mode == "J" else np.eye(n)
    F0 = float(norm(Fx))
    res = [1.0]
    status = "maxit"
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        for _ in range(maxit):
            if norm(Fx) < tol:
                status = "conv"
                break
            try:
                d = solve(B, -Fx)
            except np.linalg.LinAlgError:
                status = "stall"
                break
            if not np.all(np.isfinite(d)):
                status = "diverge"
                break
            x_new = x + d
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                status = "diverge"
                break
            s = d
            y = Fx_new - Fx
            x, Fx = x_new, Fx_new
            res.append(float(norm(Fx)) / F0)
            if norm(Fx) < tol:
                status = "conv"
                break
            denom = float(s @ s)
            if abs(denom) < 1e-20:
                status = "stall"
                break
            B = B + np.outer(y - B @ s, s) / denom
    return np.array(res), status


def broyden_ls(F, Jf, x0, b0_mode="J", maxit=200, tol=1e-8,
               beta=0.5, sigma=1e-4, a_min=1e-12):
    """Бройден с НЕМОНОТОННЫМ безпроизводным line search (в духе
    Li–Fukushima 2000). Шаг \\alpha=\\beta^i — наибольший, при котором
        ||F(x+alpha d)|| <= (1+eta_k)||F(x)|| - sigma*alpha^2*||d||^2,
    eta_k = 1/(k+1)^2 (суммируемо). Член eta_k допускает временный рост
    ||F||, что и нужно для не-спусковых квазиньютоновских направлений;
    при alpha->0 правая часть > ||F(x)||, т.е. шаг всегда находится.
    Возвращает (res_norm, status)."""
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    B = Jf(x).copy() if b0_mode == "J" else np.eye(n)
    F0 = float(norm(Fx))
    res = [1.0]
    status = "maxit"
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        for k in range(maxit):
            if norm(Fx) < tol:
                status = "conv"
                break
            try:
                d = solve(B, -Fx)
            except np.linalg.LinAlgError:
                status = "stall"
                break
            if not np.all(np.isfinite(d)):
                status = "diverge"
                break
            eta = 1.0 / (k + 1.0) ** 2
            nFx = float(norm(Fx))
            dd = float(d @ d)
            alpha = 1.0
            x_new = x + alpha * d
            Fx_new = F(x_new)
            while (not np.all(np.isfinite(Fx_new))
                   or norm(Fx_new) > (1.0 + eta) * nFx - sigma * alpha * alpha * dd):
                alpha *= beta
                if alpha < a_min:
                    break
                x_new = x + alpha * d
                Fx_new = F(x_new)
            if alpha < a_min or not np.all(np.isfinite(Fx_new)):
                status = "stall"
                break
            s = alpha * d
            y = Fx_new - Fx
            x, Fx = x_new, Fx_new
            res.append(float(norm(Fx)) / F0)
            if norm(Fx) < tol:
                status = "conv"
                break
            denom = float(s @ s)
            if abs(denom) < 1e-20:
                status = "stall"
                break
            B = B + np.outer(y - B @ s, s) / denom
    return np.array(res), status


def make_globalization_fig():
    """ЧЕСТНЫЙ результат (своя реализация, рисунки Li–Fukushima НЕ копируются).
    На далёком старте (d≈1.5) line search-глобализация УСТРАНЯЕТ blow-up
    (без неё ||F|| уходит в overflow), но на этой задаче НЕ доводит до корня:
    итерации застревают на ~7% невязки — локальный минимум merit ||F||,
    куда попадает и демпфированный Ньютон. Т.е. честно показываем «нет blow-up»,
    но НЕ «сходимость» (это было бы фабрикацией)."""
    n = 100
    maxit = 200
    tol = 1e-8
    n_seeds = 10
    d_rel = 1.5
    x_star = newton(btri_F, btri_J, -np.ones(n))

    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    variants = [
        ("без line search — blow-up (расходимость)",      "#b2182b", broyden_classic),
        ("с line search — без blow-up, но застой ($\\sim$7\\%)", "#1b7837", broyden_ls),
    ]
    print(f"\n[globalization] d≈{d_rel}, Broyden Tridiagonal n={n}")
    for label, color, solver in variants:
        rng2 = np.random.default_rng(20260616)   # одинаковые старты для обоих
        trajs = []
        n_conv = 0
        for _ in range(n_seeds):
            u = rng2.standard_normal(n)
            u /= norm(u)
            x0 = x_star + d_rel * norm(x_star) * u
            res, st = solver(btri_F, btri_J, x0, b0_mode="J",
                             maxit=maxit, tol=tol)
            trajs.append(np.clip(res, None, 1e3))
            n_conv += int(st == "conv")
        print(f"  {label:<32s}: сошлись {n_conv}/{n_seeds}")
        for t in trajs:
            ax.plot(np.arange(len(t)), t, color=color, lw=0.8, alpha=0.40)
        L = max(len(t) for t in trajs)
        M = np.full((n_seeds, L), np.nan)
        for i, t in enumerate(trajs):
            M[i, :len(t)] = t
        ax.plot(np.arange(L), np.nanmedian(M, axis=0), color=color, lw=2.4,
                label=label)
    ax.axhline(1.0, color="#444", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e3)
    ax.set_xlim(0, maxit)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|F(x_k)\|_2 \,/\, \|F(x_0)\|_2$")
    ax.set_title("Глобализация убирает blow-up, но не доводит до корня "
                 "(merit-минимум): Broyden Tridiagonal, $d\\approx1.5$")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=9.5, loc="center left")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Презентация_магистр__диссертации", "figs",
                       "fig_motiv_globalization.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    n = 100
    n_seeds = 10
    maxit = 150
    tol = 1e-8
    b0_mode = os.environ.get("B0", "J")   # "J" (faithful to BDM thm) или "I"
    rng = np.random.default_rng(20260615)

    x_star = newton(btri_F, btri_J, -np.ones(n))
    print(f"||F(x*)|| = {norm(btri_F(x_star)):.2e},  ||x*|| = {norm(x_star):.3f}")

    # относительные расстояния старта от корня: d = ||x0 - x*|| / ||x*||
    groups = [
        (0.1, r"старт близко ($d\approx0.1$): сходимость",      "#1b7837"),
        (1.5, r"старт далеко ($d\approx1.5$): расходимость",    "#b2182b"),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    print(f"\nB_0 = {b0_mode};  классический Бройден, tol={tol:g}, maxit={maxit}")
    for d_rel, label, color in groups:
        trajs = []
        n_conv = 0
        iters = []
        for _ in range(n_seeds):
            u = rng.standard_normal(n)
            u /= norm(u)
            x0 = x_star + d_rel * norm(x_star) * u
            res, status = broyden_classic(btri_F, btri_J, x0,
                                          b0_mode=b0_mode, maxit=maxit, tol=tol)
            trajs.append(np.clip(res, None, 1e3))   # клип для отображения overflow
            n_conv += int(status == "conv")
            iters.append(len(res))
        print(f"  d≈{d_rel:<4}: сошлись {n_conv}/{n_seeds}, "
              f"итераций median={int(np.median(iters))}, max={max(iters)}")
        for t in trajs:
            ax.plot(np.arange(len(t)), t, color=color, lw=0.8, alpha=0.40)
        L = max(len(t) for t in trajs)
        M = np.full((n_seeds, L), np.nan)
        for i, t in enumerate(trajs):
            M[i, :len(t)] = t
        med = np.nanmedian(M, axis=0)
        ax.plot(np.arange(L), med, color=color, lw=2.4, label=label)

    ax.axhline(1.0, color="#444", lw=0.8, ls="--")
    ax.text(maxit * 0.99, 1.35, "уровень старта", ha="right", va="bottom",
            fontsize=8, color="#444")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e3)
    ax.set_xlim(0, 80)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|F(x_k)\|_2 \,/\, \|F(x_0)\|_2$")
    ax.set_title("Классический Бройден, $B_0=J_F(x_0)$; "
                 "Broyden Tridiagonal (MGH #30), $n=100$")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=9.5, loc="center left")
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Презентация_магистр__диссертации", "figs")
    out_pdf = os.path.join(out_dir, "fig_motiv_start_sensitivity.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_pdf}")


if __name__ == "__main__":
    main()
    make_globalization_fig()
