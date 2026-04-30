"""
diag_ss_sr1_scaling.py — расширенный скейлинг радиуса локальной сходимости
SS-SR1 vs SR1 на задаче Curved Valley (n=2) по log-сетке alpha c
доверительными интервалами по случайным стартовым точкам.

Закрывает пункт P1 чеклиста «SS-SR1: 3 точек недостаточно для power-law
фита; нужно >=10 точек на log-сетке».

Постановка:
  f(x) = (alpha/2)*(x_2 - beta * x_1^2)^2 + 0.5*(x_1^2 + x_2^2),  n=2.
  Глобальный минимум x* = 0; J* = diag(1, 1+alpha) при beta=1.
  Радиус сходимости R(method) определяется бисекцией: для случайной
  направляющей u (||u||=1) ищем максимальное r такое, что starting
  point x0 = r*u приводит к ||x_k|| <= 1e-6 за <= max_iter итераций.

  Скейлинг: фиксируем beta, варьируем alpha по log-сетке (10^0.5 .. 10^2.5,
  >=12 точек). На каждой точке alpha:
    10 случайных u, бисекция R; репортим медиану и IQR.
  По медианам делаем log--log регрессию R = C*(alpha)^p и определяем p
  с bootstrap-CI 95%.

Методы: SR1, SS-SR1 (общая n-мерная реализация из diag_ss_sr1.py).

Выходы:
  fig_ss_sr1_scaling.pdf — log-log R vs alpha с IQR-полосой и фитом.
  ss_sr1_scaling.npz     — сырые радиусы.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy.linalg import norm, solve, LinAlgError


# ============================================================
#   2D обновления (точная реплика basin.ipynb)
# ============================================================

def sr1_update_2d(B, s, y):
    r = y - B @ s
    d = float(r @ s)
    if abs(d) < 1e-8 * norm(r) * norm(s):
        return B
    return B + np.outer(r, r) / d


def ss_sr1_update_2d(B, s, y, s_prev, y_prev):
    """SS-SR1 в 2D: SR1 + одна MinSecant-коррекция вдоль s_perp.
    Использует ровно одну прошлую секущую (s_prev, y_prev), как в
    оригинальной реализации basin.ipynb. Здесь u = R_{90}\hat s
    --- единичный вектор, ортогональный s.
    """
    r = y - B @ s
    d = float(r @ s)
    if abs(d) >= 1e-8 * norm(r) * norm(s):
        B = B + np.outer(r, r) / d
    if s_prev is None:
        return B
    R_past = B @ s_prev - y_prev
    s_hat = s / norm(s)
    u = np.array([-s_hat[1], s_hat[0]])
    stu = float(s_prev @ u)
    if abs(stu) > 1e-15:
        sigma = -float(u @ R_past) / stu
        B = B + sigma * np.outer(u, u)
    return B


def run_pure(prob, method, max_iter=200, tol=1e-12):
    """Точная реплика basin.run_method для 2D Curved Valley.

    Линейный поиск на psi = ||F||^2 с целью psi <= psi0*(1 - 1e-4*alpha),
    жёсткий cap шага: ||d|| <= 50 * max(||x||, 1).
    Возвращает (converged, k, x).
    """
    F = prob["F"]; x = prob["x0"].copy()
    n = x.size
    B = np.eye(n)
    s_prev = None; y_prev = None
    for k in range(max_iter):
        Fk = F(x); fn = norm(Fk)
        if fn < tol:
            return True, x
        try:
            d = solve(B, -Fk)
        except LinAlgError:
            return False, x
        if not np.all(np.isfinite(d)):
            return False, x
        nd = norm(d)
        if nd > 50.0 * max(norm(x), 1.0):
            d *= 50.0 * max(norm(x), 1.0) / nd
        a = 1.0; phi0 = fn**2
        for _ in range(30):
            xt = x + a * d
            if norm(F(xt))**2 <= phi0 * (1.0 - 1e-4 * a):
                break
            a *= 0.5
        else:
            a = 1e-4
        s = a * d; x_new = x + s
        if not np.all(np.isfinite(x_new)) or norm(x_new) > 1e8:
            return False, x_new
        y = F(x_new) - Fk
        if method == 'sr1':
            B = sr1_update_2d(B, s, y)
        elif method == 'ss_sr1':
            B = ss_sr1_update_2d(B, s, y, s_prev, y_prev)
        else:
            raise ValueError(method)
        s_prev = s.copy(); y_prev = y.copy()
        x = x_new
    return norm(F(x)) < tol, x

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


def make_curved_valley(alpha=50.0, beta=1.0):
    """2D Curved Valley в форме F(x) = grad f(x), как в basin.ipynb."""
    def F(x):
        x1, x2 = x
        c = x2 - beta*x1**2
        return np.array([x1 - 2*alpha*beta*x1*c, x2 + alpha*c])
    def J(x):
        x1, x2 = x
        c = x2 - beta*x1**2
        return np.array([[1 + 2*alpha*beta*(3*beta*x1**2 - x2), -2*alpha*beta*x1],
                         [-2*alpha*beta*x1,                      1 + alpha]])
    return dict(name=f"CV2D(a={alpha:.0f},b={beta:g})",
                n=2, F=F, J=J, x0=np.zeros(2))


def converges_from(prob, x0, method, max_iter=200, tol=1e-12):
    """Точная реплика find_radius из basin.ipynb."""
    prob = dict(prob)
    prob["x0"] = x0.copy()
    ok, _ = run_pure(prob, method, max_iter=max_iter, tol=tol)
    return ok


def radius_bisect(prob, method, u_dir, r_lo=1e-3, r_hi=6.0,
                  max_iter=200, tol=1e-12, depth=22):
    """Бисекция: ищем максимальный r такой, что метод сходится с x0 = r*u_dir.
    """
    # Если даже r_hi сходится — это потолок (как в табл. 3.1: R_max = 6).
    if converges_from(prob, r_hi*u_dir, method, max_iter=max_iter, tol=tol):
        return r_hi
    # Если r_lo не сходится — нет смысла, метод вообще не сошёлся
    if not converges_from(prob, r_lo*u_dir, method, max_iter=max_iter, tol=tol):
        return 0.0
    a, b = r_lo, r_hi
    for _ in range(depth):
        m = 0.5*(a+b)
        if converges_from(prob, m*u_dir, method, max_iter=max_iter, tol=tol):
            a = m
        else:
            b = m
    return a  # нижняя граница «сходится точно»


def main():
    rng = np.random.default_rng(20260428)
    beta = 1.0
    alphas = np.geomspace(3.0, 300.0, 12)  # 12 точек log-сетки
    seeds_per_alpha = 10
    methods = [('sr1',    'SR1',    'tab:blue'),
               ('ss_sr1', 'SS-SR1', 'tab:red')]

    R_data = {m[0]: np.zeros((len(alphas), seeds_per_alpha)) for m in methods}
    for ai, alpha in enumerate(alphas):
        prob = make_curved_valley(alpha=alpha, beta=beta)
        # фиксированный набор направлений на каждом alpha (sphere uniform)
        for si in range(seeds_per_alpha):
            theta = rng.uniform(0, 2*np.pi)
            u = np.array([np.cos(theta), np.sin(theta)])
            for mk, *_ in methods:
                R = radius_bisect(prob, mk, u, r_lo=1e-3, r_hi=6.0,
                                   max_iter=200, tol=1e-12)
                R_data[mk][ai, si] = R
        # отчёт по медиане
        med = {mk: float(np.median(R_data[mk][ai])) for mk, *_ in methods}
        print(f"alpha={alpha:7.2f}  R_SR1={med['sr1']:6.3f}  "
              f"R_SS-SR1={med['ss_sr1']:6.3f}  "
              f"ratio={med['ss_sr1']/max(med['sr1'],1e-12):.2f}")

    # ---- log-log fit + bootstrap CI на показатель ----
    def fit_power(x, y):
        # log y = log C + p log x   --> МНК
        lx, ly = np.log(x), np.log(np.maximum(y, 1e-12))
        p, c = np.polyfit(lx, ly, 1)
        return p, np.exp(c)

    def bootstrap_p(x, y, n=2000, conf=95):
        ps = []
        n_pts = len(x)
        for _ in range(n):
            idx = rng.integers(0, n_pts, size=n_pts)
            xb, yb = x[idx], y[idx]
            ps.append(fit_power(xb, yb)[0])
        ps = np.array(ps)
        lo = np.percentile(ps, (100-conf)/2)
        hi = np.percentile(ps, 100-(100-conf)/2)
        return float(np.mean(ps)), float(lo), float(hi)

    fits = {}
    for mk, mlab, mc in methods:
        med = np.median(R_data[mk], axis=1)
        # отбрасываем точки с R=0 (метод вообще не сошёлся) для фита
        mask = med > 1e-10
        p_hat, C_hat = fit_power(alphas[mask], med[mask])
        p_mean, p_lo, p_hi = bootstrap_p(alphas[mask], med[mask])
        fits[mk] = (p_hat, C_hat, p_lo, p_hi)
        print(f"{mlab}: R ~ ({alphas.min():.0f}..{alphas.max():.0f})^{p_hat:.3f}  "
              f"95%CI [{p_lo:.3f}, {p_hi:.3f}]  C={C_hat:.3g}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for mk, mlab, mc in methods:
        med = np.median(R_data[mk], axis=1)
        q25 = np.percentile(R_data[mk], 25, axis=1)
        q75 = np.percentile(R_data[mk], 75, axis=1)
        p_hat, C_hat, p_lo, p_hi = fits[mk]
        ax.fill_between(alphas, q25, q75, alpha=0.20, color=mc)
        ax.plot(alphas, med, 'o-', lw=1.6, color=mc,
                label=f"{mlab}  $p={p_hat:.3f}$  CI[{p_lo:.3f}, {p_hi:.3f}]")
        # фит
        x_fit = np.geomspace(alphas.min(), alphas.max(), 50)
        ax.plot(x_fit, C_hat*x_fit**p_hat, ls='--', color=mc, alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\alpha\beta$ (= $\alpha$, $\beta=1$)")
    ax.set_ylabel(r"медианный радиус $R$ (10 seeds)")
    ax.set_title(r"Скейлинг радиуса по $\alpha$, log-сетка из 12 точек")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    ax = axes[1]
    med_sr1   = np.median(R_data['sr1'],    axis=1)
    med_sssr1 = np.median(R_data['ss_sr1'], axis=1)
    ratio = med_sssr1 / np.maximum(med_sr1, 1e-12)
    # IQR ratio: ratio of medians (consistent with display)
    q25 = np.percentile(R_data['ss_sr1']/np.maximum(R_data['sr1'], 1e-12), 25, axis=1)
    q75 = np.percentile(R_data['ss_sr1']/np.maximum(R_data['sr1'], 1e-12), 75, axis=1)
    ax.fill_between(alphas, q25, q75, alpha=0.2, color='tab:purple')
    ax.plot(alphas, ratio, 'o-', lw=1.7, color='tab:purple',
            label='median ratio (10 seeds)')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r"$\alpha\beta$")
    ax.set_ylabel(r"$R(\mathrm{SS{-}SR1})/R(\mathrm{SR1})$")
    ax.set_title(r"Отношение радиусов")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

    p1, _, lo1, hi1 = fits['sr1']
    p2, _, lo2, hi2 = fits['ss_sr1']
    fig.suptitle(
        f"Power-law фит на 12 log-точках, beta=1: "
        f"SR1 p={p1:.3f} [{lo1:.3f}, {hi1:.3f}], "
        f"SS-SR1 p={p2:.3f} [{lo2:.3f}, {hi2:.3f}]",
        fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.94])
    out = os.path.join(THESIS_DIR, "fig_ss_sr1_scaling.pdf")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"saved: {out}")

    npz_out = os.path.join(SCRIPT_DIR, "ss_sr1_scaling.npz")
    np.savez_compressed(npz_out, alphas=alphas, beta=beta,
                         R_sr1=R_data['sr1'], R_ss_sr1=R_data['ss_sr1'],
                         fit_sr1=np.array(fits['sr1']),
                         fit_ss_sr1=np.array(fits['ss_sr1']))
    print(f"saved raw: {npz_out}")


if __name__ == "__main__":
    main()
