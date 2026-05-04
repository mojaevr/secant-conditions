"""diag_ss_conv_ci.py — статистическая версия рисунка fig:ss_conv в гл. 3.

Заменяет одно-стартовый рисунок (x_0=(2,2) для alpha=50,beta=1; x_0=(1.8,1.8)
для alpha=100,beta=1.5) на статистическую: 50 случайных направлений на той же
сфере фиксированного радиуса R_0 = ||x_0||, для каждого направления записываем
полную траекторию ||grad f||_k и кумулятивное число grad-вычислений.

Рисуем для каждой из двух конфигураций две панели:
  - (k, ||grad f||): медиана по 50 запускам + IQR-полоса 25/75 percentile,
    а также доля сошедшихся к tol=1e-12 на каждом k;
  - (#grad-вызовов, ||grad f||): то же по абсциссе времени.

Постановка обновления повторяет _solve2 / _F_cv / run_pure_cv из
diag_table31_ci.py: 2D Curved Valley, B_0 = I, Армихо-backtracking
psi=||F||^2, c1=1e-4, alpha<-alpha/2, step-cap ||d||<=50 max(||x||,1).
SR1 со skip-rule |s.r|<1e-8 ||s|| ||r||; SS-SR1 = SR1 + MS-коррекция вдоль
u = R_{90}(s_hat), одна прошлая секущая.

Выходы:
  fig_ss_sr1_conv.pdf       — заменяет существующий single-run рисунок;
  ss_sr1_conv_ci.npz        — сырые траектории и параметры.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import hypot, isfinite


# ---------- 2D обновления и solver (низкоуровневая реплика basin.ipynb) ----------

def _solve2(B00, B01, B10, B11, r0, r1):
    det = B00 * B11 - B01 * B10
    if det == 0.0 or not isfinite(det):
        return 0.0, 0.0, False
    inv = 1.0 / det
    d0 =  inv * ( B11 * r0 - B01 * r1)
    d1 =  inv * (-B10 * r0 + B00 * r1)
    if not (isfinite(d0) and isfinite(d1)):
        return 0.0, 0.0, False
    return d0, d1, True


def _F_cv(x0, x1, alpha, beta):
    c = x1 - beta * x0 * x0
    return (x0 - 2.0 * alpha * beta * x0 * c,
            x1 + alpha * c)


def trajectory_cv(x0_init, alpha, beta, method,
                  max_iter=200, tol=1e-12,
                  bt_max=30, c1_armijo=1e-4):
    """Запуск SR1/SS-SR1 на 2D Curved Valley с записью траектории.

    Возвращает dict:
      gnorm:  массив ||grad f||_k длиной (k_term+1) включая x_0 и финальную точку;
      gcount: кумулятивное число grad-вычислений на момент записи каждой точки;
      converged: bool — достигнут ли tol;
      iters:  число выполненных итераций (k_term).
    """
    x0, x1 = float(x0_init[0]), float(x0_init[1])
    B00, B01, B10, B11 = 1.0, 0.0, 0.0, 1.0
    sp0 = sp1 = yp0 = yp1 = 0.0
    has_prev = False

    F0, F1 = _F_cv(x0, x1, alpha, beta)
    gn = hypot(F0, F1)
    gnorms = [gn]
    gcounts = [1]   # один grad-вызов на старте
    cum_calls = 1
    converged = (gn < tol)
    k_done = 0

    for k in range(max_iter):
        if gn < tol:
            converged = True
            break
        d0, d1, ok = _solve2(B00, B01, B10, B11, -F0, -F1)
        if not ok:
            break
        nd = hypot(d0, d1)
        cap = 50.0 * max(hypot(x0, x1), 1.0)
        if nd > cap:
            sc = cap / nd
            d0 *= sc; d1 *= sc
        a = 1.0
        phi0 = gn * gn
        accepted = False
        for _bt in range(bt_max):
            xt0 = x0 + a * d0; xt1 = x1 + a * d1
            Ft0, Ft1 = _F_cv(xt0, xt1, alpha, beta)
            cum_calls += 1
            if Ft0*Ft0 + Ft1*Ft1 <= phi0 * (1.0 - c1_armijo * a):
                accepted = True
                break
            a *= 0.5
        if not accepted:
            a = 1e-4
        s0 = a * d0; s1 = a * d1
        xn0 = x0 + s0; xn1 = x1 + s1
        if not (isfinite(xn0) and isfinite(xn1)) or hypot(xn0, xn1) > 1e8:
            break
        Fn0, Fn1 = _F_cv(xn0, xn1, alpha, beta)
        cum_calls += 1
        gn_new = hypot(Fn0, Fn1)
        y0 = Fn0 - F0; y1 = Fn1 - F1
        # ----- SR1 update -----
        Bs0 = B00*s0 + B01*s1; Bs1 = B10*s0 + B11*s1
        r0 = y0 - Bs0; r1 = y1 - Bs1
        rs = r0*s0 + r1*s1
        nr = hypot(r0, r1); ns = hypot(s0, s1)
        if abs(rs) >= 1e-8 * nr * ns and rs != 0.0:
            inv = 1.0 / rs
            B00 += inv * r0 * r0
            B01 += inv * r0 * r1
            B10 += inv * r1 * r0
            B11 += inv * r1 * r1
        # ----- SS-SR1 extra: MS-correction along s_perp -----
        if method == 'ss_sr1' and has_prev and ns > 0.0:
            inv_ns = 1.0 / ns
            sh0 = s0 * inv_ns; sh1 = s1 * inv_ns
            u0 = -sh1; u1 = sh0
            Bp0 = B00*sp0 + B01*sp1; Bp1 = B10*sp0 + B11*sp1
            Rp0 = Bp0 - yp0; Rp1 = Bp1 - yp1
            stu = sp0*u0 + sp1*u1
            if abs(stu) > 1e-15:
                uRp = u0*Rp0 + u1*Rp1
                sigma = -uRp / stu
                B00 += sigma * u0 * u0
                B01 += sigma * u0 * u1
                B10 += sigma * u1 * u0
                B11 += sigma * u1 * u1
        sp0, sp1, yp0, yp1 = s0, s1, y0, y1
        has_prev = True
        x0, x1 = xn0, xn1
        F0, F1 = Fn0, Fn1
        gn = gn_new
        gnorms.append(gn)
        gcounts.append(cum_calls)
        k_done = k + 1
        if gn < tol:
            converged = True
            break

    return dict(gnorm=np.asarray(gnorms),
                gcount=np.asarray(gcounts),
                converged=converged,
                iters=k_done)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


def aggregate(trajs, max_k):
    """Pad ||grad|| trajectories to common length max_k+1.
    Полит pad'инга: продолжаем последним значением (если метод сошёлся,
    значение остаётся ~tol; если расходится, остаётся последний наблюдаемый).
    """
    n = len(trajs)
    G = np.full((n, max_k+1), np.nan)
    for i, tr in enumerate(trajs):
        L = len(tr['gnorm'])
        G[i, :L] = tr['gnorm']
        if L < max_k+1:
            G[i, L:] = tr['gnorm'][-1]
    return G


def main():
    rng = np.random.default_rng(20260503)
    n_dirs = 50
    max_iter = 200
    tol = 1e-12

    configs = [
        dict(alpha=50.0,  beta=1.0,  R0=2.0*np.sqrt(2),
             label=r"Curved Valley, $\alpha=50,\ \beta=1{,}0$, $\|x_0\|=2\sqrt{2}$"),
        dict(alpha=100.0, beta=1.5,  R0=1.8*np.sqrt(2),
             label=r"Curved Valley, $\alpha=100,\ \beta=1{,}5$, $\|x_0\|=1{,}8\sqrt{2}$"),
    ]
    methods = [
        ('sr1',    'SR1',    'tab:blue', '--'),
        ('ss_sr1', 'SS-SR1', 'tab:red',  '-'),
    ]

    # общие направления для всех конфигураций (paired по направлениям)
    thetas = rng.uniform(0, 2*np.pi, size=n_dirs)
    U = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

    results = {}  # (cfg_idx, method) -> list[dict]
    for ci, cfg in enumerate(configs):
        for mk, *_ in methods:
            trajs = []
            for di in range(n_dirs):
                x0 = cfg['R0'] * U[di]
                trajs.append(trajectory_cv(x0, cfg['alpha'], cfg['beta'], mk,
                                           max_iter=max_iter, tol=tol))
            n_conv = sum(1 for t in trajs if t['converged'])
            print(f"cfg={ci} alpha={cfg['alpha']:.0f} beta={cfg['beta']:.1f} "
                  f"method={mk}: {n_conv}/{n_dirs} сошлось")
            results[(ci, mk)] = trajs

    # ---------- Figure: 2x2 (spaghetti + median) ----------
    # Для бимодальных распределений (часть запусков сходится, часть
    # стагнирует) IQR-полоса растягивается на ~14 порядков и теряет
    # смысл «типичного диапазона». Вместо неё — все 50 траекторий
    # тонкими полупрозрачными линиями (spaghetti) + жирная медиана.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for ci, cfg in enumerate(configs):
        ax_iter = axes[ci, 0]
        ax_call = axes[ci, 1]
        for mk, mlab, mc, mls in methods:
            trajs = results[(ci, mk)]
            n_conv = sum(1 for t in trajs if t['converged'])
            G = aggregate(trajs, max_iter)  # (n_dirs, max_iter+1)
            ks = np.arange(G.shape[1])
            med = np.nanmedian(G, axis=0)
            label = f"{mlab} ({n_conv}/{n_dirs} сошлось)"
            # spaghetti: каждая траектория тонкой полупрозрачной линией
            for i in range(G.shape[0]):
                ax_iter.plot(ks, G[i], color=mc, lw=0.5, alpha=0.18)
            # медиана поверх
            ax_iter.plot(ks, med, color=mc, lw=2.2, ls=mls, label=label,
                         zorder=5)

            # абсцисса = #grad-вызовов: интерполируем на общую сетку
            max_calls = max(t['gcount'][-1] for t in trajs)
            grid = np.linspace(1, max_calls, 400)
            stacked = np.full((len(trajs), len(grid)), np.nan)
            for ti, t in enumerate(trajs):
                gc = t['gcount']
                gn = t['gnorm']
                if len(gc) >= 2:
                    interp = np.interp(grid, gc, gn,
                                       left=gn[0], right=gn[-1])
                    stacked[ti] = interp
                else:
                    stacked[ti] = gn[0]
            med_c = np.nanmedian(stacked, axis=0)
            for i in range(stacked.shape[0]):
                ax_call.plot(grid, stacked[i], color=mc, lw=0.5, alpha=0.18)
            ax_call.plot(grid, med_c, color=mc, lw=2.2, ls=mls, label=label,
                         zorder=5)

        for ax in (ax_iter, ax_call):
            ax.set_yscale('log')
            ax.axhline(tol, color='gray', ls=':', lw=0.8, alpha=0.7)
            ax.grid(True, which='both', alpha=0.25)
        ax_iter.set_xlabel(r"итерация $k$")
        ax_iter.set_ylabel(r"$\|\nabla f(x_k)\|$")
        ax_iter.set_title(cfg['label'], fontsize=10)
        ax_iter.legend(fontsize=8, loc='upper right')
        ax_call.set_xlabel(r"число вызовов $\nabla f$")
        ax_call.set_ylabel(r"$\|\nabla f(x_k)\|$")
        ax_call.set_title(cfg['label'], fontsize=10)
        ax_call.legend(fontsize=8, loc='upper right')

    fig.suptitle(
        f"Сходимость SR1 и SS-SR1 на Curved Valley: {n_dirs} случайных "
        r"направлений $\|x_0\|=\mathrm{const}$ (тонкие линии) + медиана",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(THESIS_DIR, "fig_ss_sr1_conv.pdf")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"\nsaved: {out}")

    # ---------- сырые ----------
    npz_out = os.path.join(SCRIPT_DIR, "ss_sr1_conv_ci.npz")
    save_dict = dict(n_dirs=n_dirs, max_iter=max_iter, tol=tol, U=U)
    for ci, cfg in enumerate(configs):
        for mk, *_ in methods:
            G = aggregate(results[(ci, mk)], max_iter)
            save_dict[f"cfg{ci}_{mk}_gnorm"] = G
            save_dict[f"cfg{ci}_{mk}_converged"] = np.array(
                [t['converged'] for t in results[(ci, mk)]])
            # gcount траектории разной длины — сохраняем как объект-массив
            save_dict[f"cfg{ci}_{mk}_gcount"] = np.array(
                [t['gcount'] for t in results[(ci, mk)]], dtype=object)
        save_dict[f"cfg{ci}_alpha"] = cfg['alpha']
        save_dict[f"cfg{ci}_beta"]  = cfg['beta']
        save_dict[f"cfg{ci}_R0"]    = cfg['R0']
    np.savez_compressed(npz_out, **save_dict)
    print(f"saved raw: {npz_out}")


if __name__ == "__main__":
    main()
