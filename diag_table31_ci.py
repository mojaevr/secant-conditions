"""
diag_table31_ci.py — bootstrap-CI к радиусам R(SR1) и R(SS-SR1) в табл. 3.1.

Закрывает пункт P2 «Remark / обсуждение» чеклиста: при α=100, β=2{,}0
табличное значение R(SR1)=2{,}97 \emph{больше}, чем R(SR1)=2{,}20 при
β=1{,}5. Чтобы понять, эффект ли это реальный (немонотонная зависимость
радиуса от кривизны β при фиксированной жёсткости α) или шум бисекции
с N=10 направлений, расширяем сетку до 50 направлений и считаем
bootstrap-CI 95\% к медиане радиуса.

Постановка повторяет diag_ss_sr1_scaling.py:
  - 2D Curved Valley (как в basin.ipynb), F(x) = grad f(x).
  - линейный поиск на psi = ||F||^2, step-cap ||d|| <= 50 max(||x||,1).
  - бисекция R с потолком r_max=6 и точностью depth=22.

Сетка: alpha=100, beta in {0.5, 1.0, 1.5, 2.0} — те же режимы, что в
табл. 3.1 (исключая тривиальный beta=0). 50 случайных направлений,
seed=20260430.

Выходы:
  fig_table31_ci.pdf    — точки медиан + 95% CI + IQR-полоса.
  table31_ci.npz        — сырые радиусы и параметры bootstrap.
  stdout: таблица «beta | R(SR1) median [95% CI] | R(SS-SR1) median [95% CI] | ratio».
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import hypot, isfinite


# ---------- 2D обновления и solver (низкоуровневая реплика basin.ipynb) ----------
# Все операции инлайнятся в скалярах, чтобы избегать numpy-оверхеда на 2x2.

def _solve2(B00, B01, B10, B11, r0, r1):
    """Решение B d = r для 2x2; возвращает (d0, d1, ok)."""
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


def run_pure_cv(x0_init, alpha, beta, method, max_iter=200, tol=1e-12):
    """2D solver на Curved Valley с инлайнингом обновлений SR1/SS-SR1.

    method: 'sr1' или 'ss_sr1'.
    Возвращает True, если ||F||<tol достигнуто за <= max_iter.
    """
    x0, x1 = float(x0_init[0]), float(x0_init[1])
    B00, B01, B10, B11 = 1.0, 0.0, 0.0, 1.0
    sp0 = sp1 = yp0 = yp1 = 0.0
    has_prev = False
    for _ in range(max_iter):
        F0, F1 = _F_cv(x0, x1, alpha, beta)
        fn = hypot(F0, F1)
        if fn < tol:
            return True
        d0, d1, ok = _solve2(B00, B01, B10, B11, -F0, -F1)
        if not ok:
            return False
        nd = hypot(d0, d1)
        cap = 50.0 * max(hypot(x0, x1), 1.0)
        if nd > cap:
            sc = cap / nd
            d0 *= sc; d1 *= sc
        a = 1.0; phi0 = fn * fn
        # Armijo-like backtracking on psi=||F||^2
        accepted = False
        for _bt in range(30):
            xt0 = x0 + a * d0; xt1 = x1 + a * d1
            Ft0, Ft1 = _F_cv(xt0, xt1, alpha, beta)
            if Ft0*Ft0 + Ft1*Ft1 <= phi0 * (1.0 - 1e-4 * a):
                accepted = True
                break
            a *= 0.5
        if not accepted:
            a = 1e-4
        s0 = a * d0; s1 = a * d1
        xn0 = x0 + s0; xn1 = x1 + s1
        if not (isfinite(xn0) and isfinite(xn1)) or hypot(xn0, xn1) > 1e8:
            return False
        Fn0, Fn1 = _F_cv(xn0, xn1, alpha, beta)
        y0 = Fn0 - F0; y1 = Fn1 - F1
        # ----- SR1 update -----
        # r = y - B s
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
            # u = R_{90}(s_hat) = (-s_hat1, s_hat0)
            inv_ns = 1.0 / ns
            sh0 = s0 * inv_ns; sh1 = s1 * inv_ns
            u0 = -sh1; u1 = sh0
            # R_past = B s_prev - y_prev
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
    F0, F1 = _F_cv(x0, x1, alpha, beta)
    return hypot(F0, F1) < tol


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


def converges_from_cv(x0, alpha, beta, method, max_iter=200, tol=1e-12):
    return run_pure_cv(x0, alpha, beta, method, max_iter=max_iter, tol=tol)


def radius_bisect_cv(alpha, beta, method, u_dir,
                     r_lo=1e-3, r_hi=6.0,
                     max_iter=200, tol=1e-12, depth=18):
    if converges_from_cv(r_hi*u_dir, alpha, beta, method,
                         max_iter=max_iter, tol=tol):
        return r_hi
    if not converges_from_cv(r_lo*u_dir, alpha, beta, method,
                              max_iter=max_iter, tol=tol):
        return 0.0
    a, b = r_lo, r_hi
    for _ in range(depth):
        m = 0.5*(a+b)
        if converges_from_cv(m*u_dir, alpha, beta, method,
                              max_iter=max_iter, tol=tol):
            a = m
        else:
            b = m
    return a


def bootstrap_median_ci(values, n_boot=2000, conf=95, rng=None):
    """Bootstrap-CI для медианы выборки (по resampling с заменой)."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    n = len(values)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        meds[b] = np.median(values[idx])
    lo = np.percentile(meds, (100 - conf) / 2)
    hi = np.percentile(meds, 100 - (100 - conf) / 2)
    return float(np.median(values)), float(lo), float(hi)


def main():
    rng = np.random.default_rng(20260430)
    alpha = 100.0
    betas = [0.5, 1.0, 1.5, 2.0]
    n_dirs = 50
    methods = [('sr1',    'SR1',    'tab:blue'),
               ('ss_sr1', 'SS-SR1', 'tab:red')]

    # фиксируем общий набор направлений для всех beta — чтобы сравнение
    # между beta производилось при одних и тех же u_i (paired design).
    thetas = rng.uniform(0, 2*np.pi, size=n_dirs)
    U = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

    R_data = {m[0]: np.zeros((len(betas), n_dirs)) for m in methods}
    for bi, beta in enumerate(betas):
        for di in range(n_dirs):
            u = U[di]
            for mk, *_ in methods:
                R_data[mk][bi, di] = radius_bisect_cv(
                    alpha, beta, mk, u, r_lo=1e-3, r_hi=6.0,
                    max_iter=200, tol=1e-12, depth=18)
        print(f"beta={beta:.1f}: SR1 median={np.median(R_data['sr1'][bi]):.3f}, "
              f"SS-SR1 median={np.median(R_data['ss_sr1'][bi]):.3f}",
              flush=True)

    # bootstrap-CI на медиану и на разность медиан (paired)
    boot_rng = np.random.default_rng(20260430)

    print("\n=== Bootstrap-CI 95% ===")
    print(f"{'beta':>5} | {'R(SR1) median [95% CI]':>30} | "
          f"{'R(SS-SR1) median [95% CI]':>32} | ratio")
    summary = {}
    for bi, beta in enumerate(betas):
        m1, lo1, hi1 = bootstrap_median_ci(R_data['sr1'][bi],
                                           n_boot=2000, rng=boot_rng)
        m2, lo2, hi2 = bootstrap_median_ci(R_data['ss_sr1'][bi],
                                           n_boot=2000, rng=boot_rng)
        ratio = m2 / max(m1, 1e-12)
        summary[beta] = dict(sr1=(m1, lo1, hi1), ss=(m2, lo2, hi2), ratio=ratio)
        print(f"{beta:5.1f} | {m1:5.2f} [{lo1:5.2f}, {hi1:5.2f}]"
              f"            | {m2:5.2f} [{lo2:5.2f}, {hi2:5.2f}]"
              f"            | {ratio:.2f}")

    # Парный тест: разность R(SR1, beta=1.5) - R(SR1, beta=2.0) по парам направлений.
    diff = R_data['sr1'][2] - R_data['sr1'][3]   # beta=1.5 vs 2.0
    n = len(diff)
    diffs_boot = np.empty(2000)
    for b in range(2000):
        idx = boot_rng.integers(0, n, size=n)
        diffs_boot[b] = np.median(diff[idx])
    d_med = float(np.median(diff))
    d_lo = float(np.percentile(diffs_boot, 2.5))
    d_hi = float(np.percentile(diffs_boot, 97.5))
    print(f"\nПарная разность R(SR1, beta=1.5) - R(SR1, beta=2.0)"
          f" по {n} направлениям:")
    print(f"  median = {d_med:+.3f}, 95% CI = [{d_lo:+.3f}, {d_hi:+.3f}]")
    contains_zero = (d_lo <= 0.0 <= d_hi)
    print(f"  CI {'СОДЕРЖИТ' if contains_zero else 'НЕ СОДЕРЖИТ'} ноль "
          f"=> различие {'не значимо' if contains_zero else 'значимо'} "
          f"на уровне 5%.")

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    bx = np.array(betas)
    for mk, mlab, mc in methods:
        key = 'sr1' if mk == 'sr1' else 'ss'
        meds = np.array([summary[b][key][0] for b in betas])
        los  = np.array([summary[b][key][1] for b in betas])
        his  = np.array([summary[b][key][2] for b in betas])
        q25 = np.percentile(R_data[mk], 25, axis=1)
        q75 = np.percentile(R_data[mk], 75, axis=1)
        ax.fill_between(bx, q25, q75, alpha=0.15, color=mc, label=f"{mlab} IQR")
        ax.errorbar(bx, meds, yerr=[meds-los, his-meds], fmt='o-', lw=1.6,
                    color=mc, capsize=4, label=f"{mlab} median + 95% CI")
    ax.axvspan(1.4, 2.1, color='gray', alpha=0.07)
    ax.set_xlabel(r"$\beta$ (кривизна), $\alpha=100$")
    ax.set_ylabel(r"радиус $R$ (потолок $R_{\max}=6$)")
    ax.set_title(r"Радиус сходимости при $\alpha=100$, "
                 f"{n_dirs} направлений")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    # boxplot по сырым R
    pos = np.arange(len(betas))
    width = 0.35
    bp1 = ax.boxplot([R_data['sr1'][bi] for bi in range(len(betas))],
                     positions=pos - width/2, widths=width, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    bp2 = ax.boxplot([R_data['ss_sr1'][bi] for bi in range(len(betas))],
                     positions=pos + width/2, widths=width, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('tab:blue'); patch.set_alpha(0.45)
    for patch in bp2['boxes']:
        patch.set_facecolor('tab:red');  patch.set_alpha(0.45)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$R$ (распределение по направлениям)")
    ax.set_title(r"Boxplot сырых радиусов")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['SR1', 'SS-SR1'],
              fontsize=9, loc='lower left')

    fig.suptitle(
        r"Bootstrap-CI к табл. 3.1: парная разность "
        r"$R(\mathrm{SR1},\beta{=}1{,}5)-R(\mathrm{SR1},\beta{=}2{,}0)$"
        + f" = {d_med:+.2f} [{d_lo:+.2f}, {d_hi:+.2f}]"
        + (" (CI содержит 0)" if contains_zero else " (CI не содержит 0)"),
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(THESIS_DIR, "fig_table31_ci.pdf")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"\nsaved: {out}")

    npz_out = os.path.join(SCRIPT_DIR, "table31_ci.npz")
    np.savez_compressed(npz_out,
                        alpha=alpha, betas=np.array(betas),
                        n_dirs=n_dirs,
                        R_sr1=R_data['sr1'], R_ss_sr1=R_data['ss_sr1'],
                        diff_med=d_med, diff_lo=d_lo, diff_hi=d_hi)
    print(f"saved raw: {npz_out}")


if __name__ == "__main__":
    main()
