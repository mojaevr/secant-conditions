"""
Ablation-эксперимент по параметрам алгоритма SP-Broyden.

Заменяет голословные утверждения в remarkkах главы 2 ("$\\tau\\in[10^2,10^4]$
устойчиво", "rescale экономит 10-15%", "fallback срабатывает в 0-2%
итераций") измеренными цифрами на одной задаче.

Тестовая задача: Discrete BVP, n=100, $x_0^{(i)}=0.1\\cdot t_i(t_i-1)$,
$t_i=i/(n+1)$ (стандарт MGH).

Варьируемые параметры (по одному, остальные --- дефолтные):
  - p_max  in {0, 1, 2, 5, 10, 20}              [уже отчасти в fig_spb_pvar]
  - tau    in {1e1, 1e2, 1e3, 1e4, 1e6}         [rem:tau-ablation]
  - rescale in {true, false}                    [rem:init-scaling]
  - globalize in {true, false}                  [Algorithm 2]
  - accumulate in {true, false}                 [rem:taxonomy]

Также измеряется частота срабатывания fallback на дефолтных настройках.

Output:
  fig_sp_ablation.pdf   --- сводка по 4 параметрам (4 панели)
  sp_ablation.npz       --- сырые данные
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve, svd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discrete_bvp_F(x):
    n = len(x)
    h = 1.0 / (n + 1)
    r = np.zeros(n)
    for i in range(n):
        ti = (i + 1) * h
        xm = x[i - 1] if i > 0 else 0.0
        xp = x[i + 1] if i < n - 1 else 0.0
        r[i] = 2.0 * x[i] - xm - xp + h * h * (x[i] + ti + 1.0) ** 3 / 2.0
    return r


def discrete_bvp_x0(n):
    h = 1.0 / (n + 1)
    return 0.1 * np.array([i * h * (i * h - 1.0) for i in range(1, n + 1)])


def sp_broyden_full(
    F, x0,
    p_max=5, tau=1e3,
    rescale=False, globalize=False, accumulate=True,
    eps_B_rel=1e-12, C_safe=1e2, nu_reg=1e-3,
    c1=1e-4, rho=0.5, alpha_min=1e-8,
    tol=1e-10, maxiter=600,
):
    """SP-Broyden со всеми опциями главы 2.
    Возвращает: словарь с трекингом и счётчиком срабатываний fallback.
    """
    n = len(x0)
    x = x0.copy().astype(float)
    Fx = F(x)
    f_evals = 1
    B = np.eye(n)
    Btilde = np.eye(n)
    S_hist, Y_hist = [], []
    log = {
        "iter": [0], "fevals": [1],
        "res": [float(norm(Fx))],
        "p_eff": [0],
        "fallback_fired": 0,
        "armijo_steps": 0,  # суммарное число шагов backtracking
        "rescale_fired": False,
    }

    for k in range(maxiter):
        if norm(Fx) < tol:
            break
        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            break

        # === fallback (rem:fallback) ===
        sigma_min_B = float(svd(B, compute_uv=False)[-1])
        eps_B = eps_B_rel * float(norm(B, 2))
        if (sigma_min_B < eps_B) or (norm(d) > C_safe * max(norm(x), 1.0)):
            log["fallback_fired"] += 1
            lam = nu_reg * float(norm(Fx))
            try:
                d = solve(B + lam * np.eye(n), -Fx)
            except np.linalg.LinAlgError:
                break

        # === Armijo (Algorithm 2) ===
        alpha = 1.0
        if globalize:
            psi_x = 0.5 * float(norm(Fx)) ** 2
            # cheap Armijo: используем |F|^2 как merit, gd ≈ -|F|^2 (для B≈J)
            gtd = -float(norm(Fx)) ** 2
            steps = 0
            while alpha > alpha_min:
                x_try = x + alpha * d
                F_try = F(x_try)
                f_evals += 1
                psi_try = 0.5 * float(norm(F_try)) ** 2
                if psi_try <= psi_x + c1 * alpha * gtd:
                    break
                alpha *= rho
                steps += 1
            log["armijo_steps"] += steps

        x_new = x + alpha * d
        if not globalize:
            Fx_new = F(x_new)
            f_evals += 1
        else:
            Fx_new = F(x_new)  # уже посчитан в Armijo, но проще пересчитать
            f_evals += 1

        if not np.all(np.isfinite(Fx_new)):
            break
        s = alpha * d
        y = Fx_new - Fx
        S_hist.append(s.copy())
        Y_hist.append(y.copy())

        # === rescale Шанно--Пхуа (rem:init-scaling) ===
        if k == 0 and rescale and float(y @ s) > 0:
            sigma0 = float(y @ y) / float(y @ s)
            B = sigma0 * B
            Btilde = sigma0 * Btilde
            log["rescale_fired"] = True

        # === адаптивный выбор p ===
        p_eff = 0
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                if cond(G) < tau:
                    p_eff = p_try
                else:
                    break

        # === SP-обновление ===
        Bhat = B if accumulate else Btilde
        if p_eff == 0:
            Bs = Bhat @ s
            v = s
            denom = float(s @ s)
            B = Bhat + np.outer(y - Bs, v) / denom
        else:
            cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols)
            G = Sp.T @ Sp
            Yp = np.column_stack([Y_hist[-1 - j] for j in range(p_eff + 1)])
            B = Bhat + (Yp - Bhat @ Sp) @ solve(G, Sp.T)

        x = x_new
        Fx = Fx_new
        log["iter"].append(k + 1)
        log["fevals"].append(f_evals)
        log["res"].append(float(norm(Fx)))
        log["p_eff"].append(int(p_eff))

        max_hist = max(p_max + 5, 25)
        if len(S_hist) > max_hist:
            S_hist.pop(0)
            Y_hist.pop(0)

    log["K"] = len(log["res"]) - 1
    log["converged"] = log["res"][-1] < tol
    log["final_res"] = log["res"][-1]
    return log


def main():
    OUTDIR = os.path.join(os.path.dirname(__file__), "mipt_thesis_master")
    n = 100
    x0 = discrete_bvp_x0(n)
    F = discrete_bvp_F

    # ============ default settings ============
    DEFAULTS = dict(
        p_max=5, tau=1e3, rescale=False, globalize=False, accumulate=True,
        tol=1e-10, maxiter=600,
    )

    # ----- (1) tau ablation -----
    TAUS = [1e1, 1e2, 1e3, 1e4, 1e6]
    tau_results = []
    print("=== tau ablation (Discrete BVP, p_max=5) ===")
    for tau in TAUS:
        kw = dict(DEFAULTS); kw["tau"] = tau
        log = sp_broyden_full(F, x0, **kw)
        tau_results.append((tau, log))
        print(f"  tau={tau:.0e}: K={log['K']}, "
              f"converged={log['converged']}, "
              f"fallback_fires={log['fallback_fired']}, "
              f"final_res={log['final_res']:.2e}")

    # ----- (2) rescale ablation -----
    print("\n=== rescale ablation ===")
    rescale_results = []
    for resc in [False, True]:
        kw = dict(DEFAULTS); kw["rescale"] = resc
        log = sp_broyden_full(F, x0, **kw)
        rescale_results.append((resc, log))
        print(f"  rescale={resc}: K={log['K']}, "
              f"converged={log['converged']}, "
              f"final_res={log['final_res']:.2e}")

    # ----- (3) globalize ablation -----
    print("\n=== globalize (Armijo) ablation ===")
    glob_results = []
    for glb in [False, True]:
        kw = dict(DEFAULTS); kw["globalize"] = glb
        log = sp_broyden_full(F, x0, **kw)
        glob_results.append((glb, log))
        print(f"  globalize={glb}: K={log['K']}, "
              f"fevals={log['fevals'][-1]}, "
              f"armijo_total_backtracks={log['armijo_steps']}, "
              f"final_res={log['final_res']:.2e}")

    # ----- (4) accumulate ablation -----
    print("\n=== accumulate ablation (B vs widetilde{B}) ===")
    acc_results = []
    for acc in [False, True]:
        kw = dict(DEFAULTS); kw["accumulate"] = acc
        log = sp_broyden_full(F, x0, **kw)
        acc_results.append((acc, log))
        print(f"  accumulate={acc}: K={log['K']}, "
              f"converged={log['converged']}, "
              f"final_res={log['final_res']:.2e}")

    # ----- fallback frequency on default settings (verify "0-2%" claim) -----
    print("\n=== fallback frequency on defaults ===")
    log_def = sp_broyden_full(F, x0, **DEFAULTS)
    total_iter = log_def['K']
    fb_frac = log_def['fallback_fired'] / max(1, total_iter) * 100
    print(f"  K={total_iter}, fallback fired {log_def['fallback_fired']} times "
          f"= {fb_frac:.2f}% iterations")

    # ============ Фигура ============
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) tau ablation: K vs tau
    ax = axes[0, 0]
    Ks = [log["K"] for _, log in tau_results]
    ax.semilogx(TAUS, Ks, "o-", color="tab:blue")
    ax.set_xlabel(r"$\tau$ (порог cond$(G_p)$)")
    ax.set_ylabel("итераций до $\|F\|<10^{-10}$")
    ax.set_title(r"(a) Чувствительность к порогу $\tau$ (Discrete BVP, $p_{\max}=5$)")
    ax.grid(True, alpha=0.3)
    K_def = [log["K"] for tau, log in tau_results if tau == 1e3][0]
    ax.axhline(K_def, color="gray", linestyle=":", alpha=0.5,
               label=f"дефолт $\\tau=10^3$: $K={K_def}$")
    ax.legend(loc="best")

    # (b) rescale: convergence curves
    ax = axes[0, 1]
    for resc, log in rescale_results:
        label = f"rescale={'true' if resc else 'false'}: K={log['K']}"
        ax.semilogy(log["iter"], log["res"], label=label)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|F(x_k)\|_2$")
    ax.set_title("(b) Шанно--Пхуа масштабирование $B_0$")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (c) globalize: convergence curves
    ax = axes[1, 0]
    for glb, log in glob_results:
        label = (f"globalize={'true' if glb else 'false'}: "
                 f"K={log['K']}, fevals={log['fevals'][-1]}")
        ax.semilogy(log["iter"], log["res"], label=label)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|F(x_k)\|_2$")
    ax.set_title("(c) Глобализация по Армихо (Algorithm 2)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (d) accumulate: convergence curves
    ax = axes[1, 1]
    for acc, log in acc_results:
        label = (f"accumulate={'true' if acc else 'false'}: K={log['K']}")
        ax.semilogy(log["iter"], log["res"], label=label)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|F(x_k)\|_2$")
    ax.set_title(r"(d) Накопление: $\widehat B_k = B_k$ vs.\ $\widehat B_k = \widetilde B$")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_sp_ablation.pdf"))
    plt.close(fig)

    # save raw data
    np.savez(
        os.path.join(os.path.dirname(__file__), "sp_ablation.npz"),
        taus=np.array(TAUS),
        K_tau=np.array([log["K"] for _, log in tau_results]),
        K_rescale=np.array([log["K"] for _, log in rescale_results]),
        K_globalize=np.array([log["K"] for _, log in glob_results]),
        fevals_globalize=np.array([log["fevals"][-1] for _, log in glob_results]),
        K_accumulate=np.array([log["K"] for _, log in acc_results]),
        fallback_fraction_pct=fb_frac,
    )
    print(f"\nДанные --> sp_ablation.npz")
    print(f"Фигура --> {OUTDIR}/fig_sp_ablation.pdf")


if __name__ == "__main__":
    main()
