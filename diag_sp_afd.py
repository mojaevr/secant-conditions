"""
Диагностические эксперименты SP-AFD для главы 4 диссертации.
Закрывает четыре пункта P1 блока «Численные эксперименты не проверяют теорию»
(VIJI / SP-AFD):
  1) log-log GAP vs T с фитом наклона   -> fig_sp_afd_gap_T.pdf
  2) тест на >=3 различных мю-сильно монотонных VI  -> fig_sp_afd_problems.pdf
  3) ||(B_k - dF(x_k)) d_k|| / ||d_k||^2 по итерациям -> fig_sp_afd_cond14.pdf
  4) r* (число FD-уточнений) по итерациям   -> fig_sp_afd_rstar.pdf

Три тестовые сильно-монотонные VI (мю = 1):
  A. Cubic monotone:        F(x) = A x + eps (x-c)^{odot 3}
                            A = U diag(1..kappa) U^T  (как в run_seeds.py)
  B. Bilinear saddle + кубика (linear-GAN-like с регуляризацией):
                            z = (x,y) in R^{2n};
                            J = [[mu I, B], [-B^T, mu I]];
                            F(z) = J z + eps (z-c)^{odot 3}.
                            Симметризация J дает мю-строгую монотонность.
  C. Smooth NCP (affine + softplus barrier):
                            F(x) = M x + q + rho * sigmoid(-x/rho).
                            sigmoid' принадлежит (0, 1/(4 rho)) — гладкий
                            барьер, smooth-аналог комплементарности x>=0.

Все три задачи параметризуются seed'ом, точное решение x* строится так,
чтобы F(x*) = 0 (для A,B — сдвигом c; для C — сдвигом q).

Для VI используем gap-индикатор g_k = ||F(x_k)||_2: при сильной монотонности
g_k <= L_1 ||x_k - x*|| и g_k >= mu ||x_k - x*||, так что log-log наклон
совпадает с таковым для ||x_k - x*||.

Воспроизводимость: NumPy 1.26+, фиксированные seed'ы.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Тестовые задачи: возвращают F(x), J(x), x*, L1, имя.
# =====================================================================
def make_cubic_monotone(n: int, kappa: float, eps: float, seed: int):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(H)
    A = Q @ np.diag(np.linspace(1.0, kappa, n)) @ Q.T
    x_star = rng.standard_normal(n) * 0.5
    Ax = A @ x_star
    c = x_star - np.cbrt(-Ax / eps)  # тогда F(x*) = 0
    L1 = 6.0 * eps  # коэффициент при кубике в J(x): 3 eps (x-c)^2

    def F(x):
        return A @ x + eps * (x - c) ** 3

    def J(x):
        return A + np.diag(3.0 * eps * (x - c) ** 2)

    return dict(F=F, J=J, x_star=x_star, L1=L1, mu=1.0, n=n,
                name=f"Cubic-Monotone (n={n}, kappa={kappa})")


def make_saddle_cubic(n: int, kappa: float, eps: float, seed: int):
    """
    z = (x,y) in R^{2n}.
    J0 = [[mu I, B], [-B^T, mu I]],   B — случайная, ||B||_op нормирована к kappa-1.
    F(z) = J0 z + eps (z - c)^{odot 3}.
    """
    rng = np.random.default_rng(seed)
    mu = 1.0
    B = rng.standard_normal((n, n))
    B = B / np.linalg.norm(B, 2) * (kappa - 1.0)  # ||B||_op = kappa-1
    J0 = np.zeros((2 * n, 2 * n))
    J0[:n, :n] = mu * np.eye(n)
    J0[:n, n:] = B
    J0[n:, :n] = -B.T
    J0[n:, n:] = mu * np.eye(n)
    z_star = rng.standard_normal(2 * n) * 0.4
    Jz = J0 @ z_star
    c = z_star - np.cbrt(-Jz / eps)
    L1 = 6.0 * eps

    def F(z):
        return J0 @ z + eps * (z - c) ** 3

    def J(z):
        return J0 + np.diag(3.0 * eps * (z - c) ** 2)

    return dict(F=F, J=J, x_star=z_star, L1=L1, mu=mu, n=2 * n,
                name=f"Bilinear-Saddle+Cubic (2n={2*n}, ||B||={kappa-1:.0f})")


def make_smooth_ncp(n: int, kappa: float, rho: float, seed: int):
    """
    F(x) = M x + q + rho * sigma'(-x/rho) , где sigma(t) = log(1+e^t) (softplus),
    sigma'(-x/rho) = 1/(1+exp(x/rho)) принадлежит (0,1).
    Якобиан: M + diag( rho * sigma''(-x/rho) * (-1/rho) ) = M - diag(s(1-s))
    с s_i = sigma'(-x_i/rho). Симметризация: M симметрична с собств.числами
    [mu, kappa]; вычитание s(1-s) <= 1/4 — задача остаётся мю/2-монотонной
    при mu > 1/2.
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(H)
    diag = np.linspace(1.0, kappa, n)
    M = Q @ np.diag(diag) @ Q.T
    x_star = rng.standard_normal(n) * 0.5

    # Стабильное вычисление sigmoid:  sig(-x/rho) = 1/(1+exp(x/rho))
    def sig_neg(x):
        # numerically stable
        out = np.empty_like(x, dtype=float)
        pos = x >= 0.0
        out[pos] = np.exp(-x[pos] / rho) / (1.0 + np.exp(-x[pos] / rho))
        np_ = ~pos
        out[np_] = 1.0 / (1.0 + np.exp(x[np_] / rho))
        return out

    def F_no_q(x):
        return M @ x + rho * sig_neg(x)

    q = -F_no_q(x_star)  # тогда F(x*) = 0

    def F(x):
        return M @ x + q + rho * sig_neg(x)

    def J(x):
        s = sig_neg(x)  # in (0,1)
        return M - np.diag(s * (1.0 - s))  # |s(1-s)| <= 1/4

    L1 = float(np.linalg.norm(M, 2) + 1.0)  # с запасом
    mu = max(0.5, 1.0 - 0.25)  # >= 0.75 при заданной нормировке M
    return dict(F=F, J=J, x_star=x_star, L1=L1, mu=mu, n=n,
                name=f"Smooth-NCP (n={n}, kappa={kappa})")


# =====================================================================
# Базовый солвер VIJI-Restart с подключаемыми правилами секущего
# обновления. Возвращает err, fcalls, cond14_ratio (по шагам), r_star.
# =====================================================================
def viji_restart(prob, x0, *, method: str, p_max: int = 3,
                 max_iter: int = 200, restart_every: int = 25,
                 cond_thresh: float = 1e3, fd_h: float = 1e-7,
                 tau_cond14: float = 0.5, r_max: int = 6,
                 beta_floor: float = 0.1, tol: float = 1e-12):
    """
    method ∈ {'broyden','sp','sp_afd','sp_afd_adaptive'}.
      'broyden':         v_k = s_k / ||s_k||^2 (классический Бройден)
      'sp':              v_k = S_p (S_p^T S_p)^{-1} e_last
      'sp_afd':          'sp' + ровно одна FD-секанта вдоль d_k (как run_seeds.py)
      'sp_afd_adaptive': inner refinement-loop по cond14
                         ( r* итераций FD-уточнений до выполнения условия ).
    """
    F, J, x_star, L1 = prob["F"], prob["J"], prob["x_star"], prob["L1"]
    n = x0.size
    x = x0.copy()
    B = np.eye(n)
    Sp = np.zeros((n, 0))
    Yp = np.zeros((n, 0))
    err = [np.linalg.norm(x - x_star)]
    fres = [np.linalg.norm(F(x))]  # gap-индикатор
    fc = [1]
    cond14 = []        # delta_k = ||(B - J(x))d|| / ||d||  (по шагам)
    step_norm = []     # ||d_k||  (для порога L_1/2 * ||d_k||)
    rstar = []         # по шагам
    f_curr = F(x)
    d_prev = np.zeros(n)

    def sp_secant(Sp, Yp):
        G = Sp.T @ Sp
        e_last = np.zeros(Sp.shape[1])
        e_last[-1] = 1.0
        try:
            return Sp @ np.linalg.solve(G, e_last)
        except np.linalg.LinAlgError:
            s = Sp[:, -1]
            return s / max(np.dot(s, s), 1e-30)

    for k in range(max_iter):
        beta_k = max(beta_floor, L1 * np.linalg.norm(d_prev))
        try:
            d = -np.linalg.solve(B + beta_k * np.eye(n), f_curr)
        except np.linalg.LinAlgError:
            break

        # cond14-индикаторы для базового шага d перед обновлением
        Jx = J(x)
        d_n = max(np.linalg.norm(d), 1e-30)
        cond14.append(np.linalg.norm((B - Jx) @ d) / d_n)  # = delta_k
        step_norm.append(d_n)

        # Armijo backtracking на ψ = ||F||^2/2
        psi = 0.5 * np.dot(f_curr, f_curr)
        alpha = 1.0
        for _ in range(20):
            x_try = x + alpha * d
            f_try = F(x_try)
            if not np.all(np.isfinite(f_try)):
                alpha *= 0.5; continue
            if 0.5 * np.dot(f_try, f_try) <= psi - 1e-4 * alpha * np.dot(f_curr, f_curr):
                break
            alpha *= 0.5
        d_used = alpha * d
        x_new = x + d_used
        f_new = F(x_new)
        fc.append(fc[-1] + 1)
        s_step = x_new - x
        y_step = f_new - f_curr

        # ---- секущее обновление ---------------------------------
        if method == 'broyden':
            v = s_step / max(np.dot(s_step, s_step), 1e-30)
            B = B + np.outer(y_step - B @ s_step, v)
            rstar.append(0)
        else:
            Sp = np.column_stack([Sp, s_step])[:, -p_max:] if Sp.size else s_step[:, None]
            Yp = np.column_stack([Yp, y_step])[:, -p_max:] if Yp.size else y_step[:, None]
            while Sp.shape[1] > 1 and np.linalg.cond(Sp.T @ Sp) > cond_thresh:
                Sp = Sp[:, 1:]; Yp = Yp[:, 1:]
            v = sp_secant(Sp, Yp)
            B = B + np.outer(y_step - B @ s_step, v)

            if method == 'sp':
                rstar.append(0)
            elif method == 'sp_afd':
                # ровно один FD-уточняющий шаг вдоль d_used
                u = d_used / max(np.linalg.norm(d_used), 1e-30)
                h = fd_h * max(1.0, np.linalg.norm(x_new))
                f_plus = F(x_new + h * u); fc[-1] += 1
                y_fd = (f_plus - f_new) / h
                Sp_fd = np.column_stack([Sp, u])[:, -p_max:]
                Yp_fd = np.column_stack([Yp, y_fd])[:, -p_max:]
                try:
                    v_fd = sp_secant(Sp_fd, Yp_fd)
                    B = B + np.outer(y_fd - B @ u, v_fd)
                    Sp, Yp = Sp_fd, Yp_fd
                except np.linalg.LinAlgError:
                    pass
                rstar.append(1)
            elif method == 'sp_afd_adaptive':
                # триггер: ||(B - J(x_new)) u||/||u|| <= tau * L1.
                # вместо J используем FD-аппроксимацию y_fd.
                rs = 0
                B_test = B
                Sp_loop, Yp_loop = Sp, Yp
                for _refine in range(r_max):
                    # candidate-направление, которое будет использовано на след. итерации
                    try:
                        d_next = -np.linalg.solve(B_test + beta_k * np.eye(n), f_new)
                    except np.linalg.LinAlgError:
                        break
                    u = d_next / max(np.linalg.norm(d_next), 1e-30)
                    h = fd_h * max(1.0, np.linalg.norm(x_new))
                    f_plus = F(x_new + h * u); fc[-1] += 1
                    y_fd = (f_plus - f_new) / h
                    rs += 1
                    residual = np.linalg.norm(y_fd - B_test @ u)
                    # обновим B этой парой (всегда — улучшает приближение)
                    Sp_loop = np.column_stack([Sp_loop, u])[:, -p_max:]
                    Yp_loop = np.column_stack([Yp_loop, y_fd])[:, -p_max:]
                    while Sp_loop.shape[1] > 1 and np.linalg.cond(Sp_loop.T @ Sp_loop) > cond_thresh:
                        Sp_loop = Sp_loop[:, 1:]; Yp_loop = Yp_loop[:, 1:]
                    try:
                        v_fd = sp_secant(Sp_loop, Yp_loop)
                        B_test = B_test + np.outer(y_fd - B_test @ u, v_fd)
                    except np.linalg.LinAlgError:
                        break
                    # триггер cond14: residual <= tau * L1
                    if residual <= tau_cond14 * L1:
                        break
                B = B_test
                Sp, Yp = Sp_loop, Yp_loop
                rstar.append(rs)
            else:
                raise ValueError(method)

        if (k + 1) % restart_every == 0:
            Sp = np.zeros((n, 0)); Yp = np.zeros((n, 0))

        x, f_curr, d_prev = x_new, f_new, d_used
        err.append(np.linalg.norm(x - x_star))
        fres.append(np.linalg.norm(f_curr))
        if err[-1] < tol or fres[-1] < tol:
            break

    return dict(err=np.array(err), fres=np.array(fres),
                fc=np.array(fc[:len(err)]),
                cond14=np.array(cond14), step_norm=np.array(step_norm),
                rstar=np.array(rstar))


# =====================================================================
# Эксперимент 1: log-log GAP vs T + фит наклона в pre-asymptotic окне.
#
# Теоретические асимптотики Agafonov2024VIJI:
#   VIQA-Broyden:  RES(x_T) = O(L_1 D^2 / T)        ~ T^{-1}
#   SP-AFD:        RES(x_T) = O(L_1 D^3 / T^{3/2})  ~ T^{-3/2}
# суть «глобальные» оценки и реализуются в pre-asymptotic режиме
# (пока ||x_k - x*|| не скатилось настолько, чтобы локальная Q-сверх-
# линейность ускорила метод сильнее любого степенного закона). На
# гладких сильно монотонных задачах сверхлинейность включается уже
# к ~25-й итерации, поэтому фит slope'а (a = d log T / d log(1/eps))
# делаем в окне eps in [10^{0}, 10^{-3}] — соответствующем первым
# 5..25 итерациям.
# =====================================================================
def exp_gap_T(out_dir: str):
    methods = [('broyden', 'VIQA-Broyden', '#888888', '-'),
               ('sp', 'VIJI + SP-Broyden', '#3060c0', '-'),
               ('sp_afd', 'VIJI + SP-AFD', '#d03030', '-')]
    SEEDS = list(range(8))
    EPS_GRID = np.geomspace(1.0, 1e-8, 25)
    EPS_FIT_LO, EPS_FIT_HI = 1e-3, 1.0  # pre-asymptotic окно для slope-фита

    # один прогон на seed=0 для левой панели
    prob = make_cubic_monotone(n=30, kappa=20.0, eps=1.5, seed=0)
    rng = np.random.default_rng(1000)
    x0 = prob["x_star"] + rng.standard_normal(prob["n"]) * 0.3
    runs_main = {m: viji_restart(prob, x0, method=m, max_iter=120, tol=1e-10)
                 for m, *_ in methods}

    # сбор T(eps) по seeds для правой панели
    T_eps = {m: [] for m, *_ in methods}
    for seed in SEEDS:
        prob_s = make_cubic_monotone(n=30, kappa=20.0, eps=1.5, seed=seed)
        rng_s = np.random.default_rng(1000 + seed)
        x0_s = prob_s["x_star"] + rng_s.standard_normal(prob_s["n"]) * 0.3
        for m, *_ in methods:
            r = viji_restart(prob_s, x0_s, method=m, max_iter=120, tol=1e-10)
            g = np.minimum.accumulate(r['fres'])  # best-so-far
            T_per_eps = []
            for eps in EPS_GRID:
                idx = np.where(g <= eps)[0]
                T_per_eps.append(int(idx[0]) + 1 if idx.size else np.nan)
            T_eps[m].append(T_per_eps)
    T_eps = {m: np.array(v) for m, v in T_eps.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # === Левая: типичная траектория, log-log ===
    Tmax = 0
    for m, lab, col, ls in methods:
        g = np.minimum.accumulate(runs_main[m]['fres'])
        T = np.arange(1, g.size + 1)
        Tmax = max(Tmax, T[-1])
        axes[0].loglog(T, np.maximum(g, 1e-12), color=col, ls=ls, lw=1.8, label=lab)
    Tref = np.geomspace(2.0, Tmax, 50)
    axes[0].loglog(Tref, 50 * Tref ** (-1.0), 'k:', lw=1.0, alpha=0.6,
                   label=r'эталон $\propto T^{-1}$')
    axes[0].loglog(Tref, 50 * Tref ** (-1.5), 'k-.', lw=1.0, alpha=0.6,
                   label=r'эталон $\propto T^{-3/2}$')
    axes[0].set_xlabel(r'$T$ (число итераций)')
    axes[0].set_ylabel(r'$\min_{j \leq k}\|F(x_j)\|_2$')
    axes[0].set_title('Сходимость в log-log (cubic-monotone, seed=0)')
    axes[0].grid(which='both', alpha=0.25)
    axes[0].legend(fontsize=8.5, loc='lower left')

    # === Правая: T(eps) vs -log eps, slope = 1/alpha в pre-asymptotic окне ===
    fits = []
    for m, lab, col, ls in methods:
        T_arr = T_eps[m]
        med = np.nanmedian(T_arr, axis=0)
        q25 = np.nanquantile(T_arr, 0.25, axis=0)
        q75 = np.nanquantile(T_arr, 0.75, axis=0)
        x_axis = -np.log10(EPS_GRID)
        ok = np.isfinite(med) & (med > 0)
        eps_mask = (EPS_GRID >= EPS_FIT_LO) & (EPS_GRID <= EPS_FIT_HI) & ok
        if eps_mask.sum() >= 4:
            slope, intercept = np.polyfit(np.log(1.0 / EPS_GRID[eps_mask]),
                                          np.log(med[eps_mask]), 1)
        else:
            slope, intercept = np.nan, 0.0
        alpha_eff = 1.0 / slope if (np.isfinite(slope) and slope > 0) else np.nan
        fits.append((m, lab, col, slope, alpha_eff))
        axes[1].errorbar(x_axis[ok], med[ok],
                         yerr=[med[ok] - q25[ok], q75[ok] - med[ok]],
                         fmt='o-', color=col, lw=1.4, ms=4, alpha=0.9,
                         label=fr'{lab}: $a={slope:.2f}$ ($\alpha_{{\mathrm{{eff}}}}\!\approx\!{alpha_eff:.1f}$)')
        # пунктирная подгонка в окне
        if np.isfinite(slope):
            xx = np.linspace(-np.log10(EPS_FIT_HI), -np.log10(EPS_FIT_LO), 2)
            yy = np.exp(intercept + slope * (xx * np.log(10)))
            axes[1].plot(xx, yy, color=col, ls='--', lw=1.0, alpha=0.6)
    # эталоны: a=1 (=> alpha=1) и a=2/3 (=> alpha=3/2). Привязка к точке (0, ~5)
    x_ref = np.linspace(0, 8, 50)
    axes[1].plot(x_ref, 5 * 10 ** (1.0 * x_ref / 1.0), 'k:', lw=1.0, alpha=0.5,
                 label=r'эталон $a{=}1$  ($\alpha{=}1$)')
    axes[1].plot(x_ref, 5 * 10 ** ((2.0 / 3.0) * x_ref), 'k-.', lw=1.0, alpha=0.5,
                 label=r'эталон $a{=}2/3$  ($\alpha{=}3/2$)')
    axes[1].set_yscale('log')
    axes[1].set_ylim(5, 5e3)
    axes[1].set_xlim(-0.2, 8.5)
    axes[1].axvspan(-np.log10(EPS_FIT_HI), -np.log10(EPS_FIT_LO),
                    color='#ddeedd', alpha=0.6, label='окно фита (pre-asymptotic)')
    axes[1].set_xlabel(r'$-\log_{10}\varepsilon$')
    axes[1].set_ylabel(r'$T(\varepsilon)$ — медиана по 8 seeds')
    axes[1].set_title(r'First-passage $T(\varepsilon)$ vs $\log(1/\varepsilon)$')
    axes[1].grid(which='both', alpha=0.25)
    axes[1].legend(fontsize=7.5, loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_gap_T.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_gap_T.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fits


# =====================================================================
# Эксперимент 2: три тестовые задачи.
# =====================================================================
def exp_three_problems(out_dir: str):
    cfgs = [
        ('A', make_cubic_monotone(n=30, kappa=20.0, eps=1.5, seed=0)),
        ('B', make_saddle_cubic(n=15, kappa=8.0, eps=1.0, seed=0)),
        ('C', make_smooth_ncp(n=30, kappa=20.0, rho=0.5, seed=0)),
    ]
    methods = [('broyden', 'VIQA-Broyden', '#888888'),
               ('sp', 'SP-Broyden', '#3060c0'),
               ('sp_afd', 'SP-AFD', '#d03030')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    summary = {}
    for ax, (lbl, prob) in zip(axes, cfgs):
        rng = np.random.default_rng(1000)
        x0 = prob["x_star"] + rng.standard_normal(prob["n"]) * 0.3
        for m, name, col in methods:
            r = viji_restart(prob, x0, method=m, max_iter=120, tol=1e-10)
            # best-iterate residual для гладкого графика
            g = np.minimum.accumulate(r['fres'])
            ax.semilogy(np.arange(g.size), g,
                        color=col, lw=1.7, label=name)
            summary[(lbl, m)] = (r['fres'][-1], r['fres'].size, int(r['fc'][-1]))
        ax.set_title(f'({lbl}) {prob["name"]}')
        ax.set_xlabel(r'итерация $k$')
        ax.set_ylabel(r'$\|F(x_k)\|_2$')
        ax.grid(which='both', alpha=0.25)
        ax.legend(fontsize=8.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_problems.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_problems.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    return summary


# =====================================================================
# Эксперимент 3: cond14-ratio по итерациям.
# =====================================================================
def exp_cond14(out_dir: str):
    """
    Строим эмпирическую проверку eq:cond14:
        delta_k := ||(B_k - J(x_k)) d_k|| / ||d_k||  <=  (L_1/2) ||d_k||
    Левая панель: delta_k и порог tau*L_1*||d_k|| (tau=1/2) как функции k.
    Правая панель: нормированное отношение rho_k := delta_k / (tau*L_1*||d_k||);
    cond14 выполнено iff rho_k <= 1.
    Сравниваем VIQA-Broyden vs SP-Broyden vs SP-AFD-adaptive (последний должен
    лежать ниже порога благодаря refinement-loop).
    """
    prob = make_cubic_monotone(n=30, kappa=20.0, eps=1.5, seed=0)
    rng = np.random.default_rng(1000)
    x0 = prob["x_star"] + rng.standard_normal(prob["n"]) * 0.3
    L1 = prob["L1"]
    tau = 0.5
    methods = [('broyden', 'VIQA-Broyden', '#888888'),
               ('sp', 'SP-Broyden', '#3060c0'),
               ('sp_afd_adaptive', 'SP-AFD (adaptive)', '#d03030')]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for m, name, col in methods:
        r = viji_restart(prob, x0, method=m, max_iter=60, tol=1e-9,
                         tau_cond14=tau, r_max=8)
        delta = r['cond14']
        d_norm = r['step_norm']
        K = np.arange(delta.size)
        thresh = tau * L1 * d_norm
        # активная зона: пока ||d_k|| > 1e-7 (избегаем численного floor)
        active = d_norm > 1e-7
        rho = delta / np.maximum(thresh, 1e-30)
        # левая: delta_k и порог
        axes[0].semilogy(K[active], delta[active], color=col, lw=1.6, label=name)
        axes[0].semilogy(K[active], thresh[active], color=col, ls=':', lw=1.0,
                         alpha=0.5)
        # правая: нормированное отношение
        axes[1].semilogy(K[active], rho[active], color=col, lw=1.6, label=name)
    axes[0].set_xlabel(r'итерация $k$')
    axes[0].set_ylabel(r'$\delta_k = \|(B_k - \nabla F(x_k))\,d_k\| / \|d_k\|$')
    axes[0].set_title(r'$\delta_k$ vs порог $\tau L_1 \|d_k\|$ (пунктир, $\tau{=}1/2$)')
    axes[0].grid(which='both', alpha=0.25)
    axes[0].legend(loc='lower left', fontsize=9.0)

    axes[1].axhline(1.0, color='k', ls='--', lw=1.0,
                    label=r'порог cond14: $\rho_k \leq 1$')
    axes[1].set_xlabel(r'итерация $k$')
    axes[1].set_ylabel(r'$\rho_k = \delta_k / (\tau L_1 \|d_k\|)$')
    axes[1].set_title(r'Нормированный индикатор cond14')
    axes[1].grid(which='both', alpha=0.25)
    axes[1].legend(loc='upper left', fontsize=9.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_cond14.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_cond14.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)


# =====================================================================
# Эксперимент 4: r* по итерациям (адаптивный SP-AFD).
# =====================================================================
def exp_rstar(out_dir: str):
    cfgs = [
        ('A', make_cubic_monotone(n=30, kappa=20.0, eps=1.5, seed=0), '#3060c0'),
        ('B', make_saddle_cubic(n=15, kappa=8.0, eps=1.0, seed=0), '#d03030'),
        ('C', make_smooth_ncp(n=30, kappa=20.0, rho=0.5, seed=0), '#2c8b34'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    rstars_all = []
    labels = []
    for lbl, prob, col in cfgs:
        rng = np.random.default_rng(1000)
        x0 = prob["x_star"] + rng.standard_normal(prob["n"]) * 0.3
        r = viji_restart(prob, x0, method='sp_afd_adaptive',
                         max_iter=80, tol=1e-10,
                         tau_cond14=0.5, r_max=6)
        K = np.arange(r['rstar'].size)
        axes[0].step(K, r['rstar'], where='post', color=col, lw=1.5,
                     label=f'({lbl}) {prob["name"].split(" (")[0]}', alpha=0.85)
        rstars_all.append(r['rstar'])
        labels.append(f'({lbl})')
    axes[0].set_xlabel(r'итерация $k$')
    axes[0].set_ylabel(r'$r^{*}_k$')
    axes[0].set_title('Число FD-уточнений по итерациям')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8.5)

    # правая панель: гистограмма r*
    rmax = max(int(rs.max()) for rs in rstars_all if rs.size) if rstars_all else 1
    bins = np.arange(0, rmax + 2) - 0.5
    width = 0.27
    centers = np.arange(0, rmax + 1)
    for i, (rs, (_, prob, col)) in enumerate(zip(rstars_all, cfgs)):
        hist, _ = np.histogram(rs, bins=bins)
        axes[1].bar(centers + (i - 1) * width, hist, width=width, color=col,
                    label=f'({labels[i]}) median={int(np.median(rs))}, mean={rs.mean():.2f}',
                    alpha=0.85, edgecolor='k')
    axes[1].set_xlabel(r'$r^{*}_k$')
    axes[1].set_ylabel('число итераций')
    axes[1].set_title(r'Распределение $r^{*}_k$ по задачам')
    axes[1].set_xticks(centers)
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_rstar.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_sp_afd_rstar.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    return [(lbl, rs.tolist()) for (lbl, _, _), rs in zip(cfgs, rstars_all)]


# =====================================================================
# Точка входа.
# =====================================================================
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mipt_thesis_master')
    os.makedirs(out_dir, exist_ok=True)

    print('=== Эксп 1: log-log GAP vs T с фитом наклона ===')
    fits = exp_gap_T(out_dir)
    for m, lab, _, slope, alpha_eff in fits:
        print(f'  {lab:25s}  a (slope) = {slope:+.3f}, alpha_eff = 1/a = {alpha_eff:.3f}')

    print('\n=== Эксп 2: три задачи (A,B,C) ===')
    summary = exp_three_problems(out_dir)
    for (lbl, m), (final_g, K, F_calls) in summary.items():
        print(f'  ({lbl}) {m:10s}  final ||F|| = {final_g:.2e}  K = {K:3d}  #F = {F_calls}')

    print('\n=== Эксп 3: cond14 по итерациям ===')
    exp_cond14(out_dir)
    print('  -> fig_sp_afd_cond14.pdf saved')

    print('\n=== Эксп 4: r* по итерациям ===')
    rs_summary = exp_rstar(out_dir)
    for lbl, rs in rs_summary:
        rs_arr = np.array(rs)
        print(f'  ({lbl}) median r*={int(np.median(rs_arr))}, mean r*={rs_arr.mean():.2f}, '
              f'max r*={int(rs_arr.max())}, n_iter={rs_arr.size}')

    print('\nDone.')


if __name__ == '__main__':
    main()
