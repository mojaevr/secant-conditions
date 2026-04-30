"""
diag_anderson.py — корректная реализация Anderson(m, beta) для системы
F(x) = 0 (mu-сильно монотонная Lipschitz-VI) и сравнение с SP-Broyden /
SP-AFD на тех же трёх классах VI, что в diag_sp_afd.py.

Закрывает пункт P1 чеклиста «Anderson baseline (нестандартная реализация
в tezisy.ipynb может давать артефактное расхождение)».

Эталонная схема Walker--Ni (2011), форма Type-II с relaxation beta:
  Положим g(x) = x - F(x). Тогда фиксированная точка g совпадает с
  корнем F. Anderson minimizes невязку f_k = g(x_k) - x_k = -F(x_k):
    Хранится окно последних m_k = min(m, k) значений
        Delta_x_i = x_{i+1} - x_i,  i = k-m_k..k-1,
        Delta_f_i = f_{i+1} - f_i,  i = k-m_k..k-1.
    Коэффициенты gamma_k решают
        min_gamma ||f_k - Delta_F_k gamma||_2^2.
    Обновление:
        x_{k+1} = x_k + beta f_k - (Delta_X_k + beta Delta_F_k) gamma_k.

Релаксация beta in (0,1] — стандартный приём для расширения области
сходимости. При beta = 1 — оригинальная схема. При beta < 1 — частичный
шаг.

Конфигурации в эксперименте:
  m in {2, 5, 10}, beta in {0.5, 1.0}  — 6 запусков на каждой задаче.

Также сравниваем с baseline VIQA-Broyden, SP-Broyden, SP-AFD из
diag_sp_afd.py (импортируем солвер).

Выходы:
  fig_anderson_baseline.pdf  — сходимость ||F||_2 vs итерация для всех
                                Anderson-конфигураций + 3 квазиньютоновских.
  fig_anderson_summary.pdf   — bar-chart числа итераций до tol на 3 задачах.
  anderson_baseline.npz       — сырые траектории.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# импорт задач + базового солвера VIJI-Restart
import diag_sp_afd as dsa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


# ============================================================
#   Anderson(m, beta), Walker--Ni 2011
# ============================================================

def anderson_solve(F, x0, m=5, beta=1.0, tau=None, L_F=None,
                   max_iter=200, tol=1e-12,
                   damp=1e-10, x_star=None, safeguard=True,
                   eta_safe=2.0, min_beta_local=1e-3):
    """Anderson(m, beta) поверх контрактивного Picard-отображения
        g_tau(x) = x - tau F(x).
    Невязка для Anderson: f_k = g_tau(x_k) - x_k = -tau F(x_k).
    При tau < 2 mu / L_F^2 g_tau контрактивно, и базовая Picard-
    итерация x_{k+1} = g_tau(x_k) сходится линейно.

    F : вычисляет F(x).
    m, beta : глубина окна и relaxation Anderson.
    tau : шаг Picard. Если None и L_F заданo, берём tau = 1/L_F.
    safeguard : Walker--Ni residual-monitoring; при росте ||F||
        beta локально уменьшается до min_beta_local; в крайнем
        случае — Picard со step tau (чистый g_tau(x)).
    """
    n = x0.size
    x = x0.copy()
    if tau is None:
        if L_F is None:
            tau = 1.0
        else:
            tau = 1.0 / float(L_F)
    Fx = F(x)
    f_curr = -tau * Fx                  # невязка фиксированно-точечной задачи
    fc = [1]
    err = [np.linalg.norm(x - x_star)] if x_star is not None else [np.nan]
    fres = [np.linalg.norm(Fx)]         # лог по физической ||F||
    X_hist = [x.copy()]
    F_hist = [f_curr.copy()]
    n_safe_restart = 0
    for k in range(max_iter):
        if fres[-1] < tol:
            break
        m_k = min(m, len(X_hist) - 1)

        def anderson_step(beta_local):
            if m_k == 0:
                return x + beta_local * f_curr
            X_arr = np.column_stack(X_hist[-(m_k+1):])
            F_arr = np.column_stack(F_hist[-(m_k+1):])
            DX = X_arr[:, 1:] - X_arr[:, :-1]
            DF = F_arr[:, 1:] - F_arr[:, :-1]
            G = DF.T @ DF
            G = G + damp * (np.trace(G) + 1.0) * np.eye(m_k)
            try:
                gamma = np.linalg.solve(G, DF.T @ f_curr)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_k)
            return x + beta_local * f_curr - (DX + beta_local * DF) @ gamma

        beta_local = beta
        x_new = anderson_step(beta_local)
        Fx_new = F(x_new); fc[-1] += 1
        if safeguard:
            while ((not np.all(np.isfinite(Fx_new)))
                   or np.linalg.norm(Fx_new) > eta_safe * fres[-1]):
                beta_local *= 0.5
                if beta_local < min_beta_local:
                    # Жёсткий рестарт: один чистый Picard-шаг с tau,
                    # окно очищаем. Это безопасный «глобализатор».
                    X_hist = [x.copy()]
                    F_hist = [f_curr.copy()]
                    n_safe_restart += 1
                    x_new = x - tau * Fx
                    Fx_new = F(x_new); fc[-1] += 1
                    break
                x_new = anderson_step(beta_local)
                Fx_new = F(x_new); fc[-1] += 1
        if not np.all(np.isfinite(Fx_new)):
            break
        f_new = -tau * Fx_new
        # сдвиг окна
        X_hist.append(x_new.copy())
        F_hist.append(f_new.copy())
        if len(X_hist) > m + 1:
            X_hist.pop(0); F_hist.pop(0)
        x = x_new
        f_curr = f_new
        Fx = Fx_new
        if x_star is not None:
            err.append(np.linalg.norm(x - x_star))
        else:
            err.append(np.nan)
        fres.append(np.linalg.norm(Fx))
        fc.append(fc[-1])
    return dict(err=np.array(err), fres=np.array(fres),
                fc=np.array(fc[:len(err)]),
                n_safe_restart=n_safe_restart, tau=tau)


# ============================================================
#   Главный эксперимент
# ============================================================

def main():
    # Три задачи: cubic-monotone, bilinear-saddle+cubic, smooth-NCP.
    # Параметры — те же, что в diag_sp_afd.exp_problems (cohereный набор).
    seed = 7
    problems = [
        ("Cubic-Monotone",  dsa.make_cubic_monotone(n=30, kappa=20.0, eps=1.5, seed=seed)),
        ("Bilinear-Saddle", dsa.make_saddle_cubic(n=15, kappa=20.0, eps=1.5, seed=seed)),
        ("Smooth-NCP",      dsa.make_smooth_ncp(n=30, kappa=20.0, rho=0.5, seed=seed)),
    ]
    rng = np.random.default_rng(20260428)

    # Anderson configurations
    and_cfgs = [(m, beta) for m in (2, 5, 10) for beta in (0.5, 1.0)]
    qn_methods = [('broyden', 'VIQA-Broyden', '#666666', ':'),
                  ('sp',      'SP-Broyden',   '#1f77b4', '-'),
                  ('sp_afd',  'SP-AFD',       '#d62728', '-')]

    results = {}  # key = (problem_name, method_label)

    for pname, prob in problems:
        x0 = prob["x_star"] + rng.standard_normal(prob["n"]) * 0.3
        # Anderson
        # tau = 1/L_F — Picard контрактивен; relaxation beta уже задана
        # в окне Anderson.
        L_F = prob["L1"]
        for m, beta in and_cfgs:
            r = anderson_solve(prob["F"], x0, m=m, beta=beta,
                                tau=1.0/L_F, L_F=L_F,
                                max_iter=200, tol=1e-10,
                                x_star=prob["x_star"])
            label = f"Anderson(m={m}, β={beta})"
            results[(pname, label)] = r
            print(f"{pname:18s}  {label:25s}  iters={len(r['fres'])-1:4d}  "
                  f"||F||={r['fres'][-1]:.2e}")
        # QN baselines
        for mk, mlab, _, _ in qn_methods:
            r = dsa.viji_restart(prob, x0, method=mk, p_max=3,
                                  max_iter=200, restart_every=25,
                                  cond_thresh=1e3, fd_h=1e-7,
                                  tau_cond14=0.5, r_max=6,
                                  beta_floor=0.1, tol=1e-12)
            results[(pname, mlab)] = r
            print(f"{pname:18s}  {mlab:25s}  iters={len(r['fres'])-1:4d}  "
                  f"||F||={r['fres'][-1]:.2e}")

    # ------- Figure 1: convergence of ||F|| vs k -------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    cmap = plt.colormaps.get_cmap('viridis')
    and_colors = [cmap(i/len(and_cfgs)) for i in range(len(and_cfgs))]
    for ax, (pname, prob) in zip(axes, problems):
        for ci, (m, beta) in enumerate(and_cfgs):
            label = f"Anderson(m={m}, β={beta})"
            r = results[(pname, label)]
            ks = np.arange(len(r['fres']))
            ax.semilogy(ks, np.maximum(r['fres'], 1e-16),
                        lw=1.3, color=and_colors[ci], label=label,
                        ls='-' if beta == 1.0 else '--')
        for mk, mlab, mc, mls in qn_methods:
            r = results[(pname, mlab)]
            ks = np.arange(len(r['fres']))
            ax.semilogy(ks, np.maximum(r['fres'], 1e-16),
                        lw=2.0, color=mc, ls=mls, label=mlab)
        ax.set_title(pname, fontsize=11)
        ax.set_xlabel(r"итерация $k$")
        ax.set_ylabel(r"$\|F(x_k)\|_2$")
        ax.grid(True, which='both', alpha=0.25)
        ax.axhline(1e-10, color='gray', ls=':', lw=0.8)
        ax.legend(fontsize=6.5, loc='upper right')
    fig.suptitle(r"Anderson($m,\beta$) baseline vs SP-Broyden / SP-AFD на 3 классах VI", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95])
    out1 = os.path.join(THESIS_DIR, "fig_anderson_baseline.pdf")
    fig.savefig(out1, bbox_inches='tight'); plt.close(fig)
    print(f"saved: {out1}")

    # ------- Figure 2: bar chart первого достижения tol = 1e-6 -------
    target = 1e-6
    def first_pass(fres, target):
        below = np.where(fres <= target)[0]
        return int(below[0]) if below.size else None

    method_labels = [f"Anderson(m={m}, β={beta})" for m, beta in and_cfgs] + \
                    [m[1] for m in qn_methods]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    bw = 0.13
    xs = np.arange(len(problems))
    palette = and_colors + [m[2] for m in qn_methods]
    for mi, mlab in enumerate(method_labels):
        vals, hatches = [], []
        for pi, (pname, _) in enumerate(problems):
            r = results[(pname, mlab)]
            t = first_pass(r['fres'], target)
            vals.append(t if t is not None else np.nan)
        offs = (mi - len(method_labels)/2 + 0.5)*bw
        ax.bar(xs + offs, vals, bw, color=palette[mi], label=mlab,
               edgecolor='black', linewidth=0.4)
        for xi, v in zip(xs + offs, vals):
            if np.isnan(v):
                ax.text(xi, 1, '—', ha='center', va='bottom',
                        fontsize=7, color='red')
            else:
                ax.text(xi, v, f'{int(v)}', ha='center', va='bottom',
                        fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([p for p,_ in problems], fontsize=10)
    ax.set_ylabel(r"Итераций до $\|F\|\leq 10^{-6}$ (—=не достигнуто)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    fig.suptitle(r"Стоимость достижения $\|F\|\leq 10^{-6}$: Anderson($m,\beta$) "
                 r"vs SP-Broyden / SP-AFD", fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.94])
    out2 = os.path.join(THESIS_DIR, "fig_anderson_summary.pdf")
    fig.savefig(out2, bbox_inches='tight'); plt.close(fig)
    print(f"saved: {out2}")

    # raw
    save_data = {}
    for (pname, mlab), r in results.items():
        prefix = f"{pname.replace(' ','_').replace('+','_').replace('(','').replace(')','').replace(',','_').replace('=','').replace('β','b').replace('.','p')}__{mlab.replace(' ','_').replace('(','').replace(')','').replace('β','b').replace('=','').replace(',','_').replace('.','p')}"
        save_data[prefix + "__fres"] = r['fres']
        save_data[prefix + "__err"]  = r['err']
        save_data[prefix + "__iters"] = len(r['fres']) - 1
    npz_out = os.path.join(SCRIPT_DIR, "anderson_baseline.npz")
    np.savez_compressed(npz_out, **save_data)
    print(f"saved raw: {npz_out}")


if __name__ == "__main__":
    main()
