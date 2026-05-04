"""
Численная проверка теоремы о пре-асимптотической фазе SP-Broyden.

Утверждение: SP-Broyden раньше входит в режим, где выполняется
условие Денниса--Море ||E_k d_k|| / ||d_k|| <= delta*, по сравнению
с классическим Бройденом (p=0).

Операциональное определение пре-асимптотической фазы:
   K_pre(delta*) := min{ k : ||E_j d_j|| / ||d_j|| <= delta* для всех j >= k }

Дополнительные диагностики:
  - nu_k = ||x_{k+1} - x*|| / ||x_k - x*||           (Q-линейный коэффициент)
  - ||E_k d_k|| / ||d_k||                            (Денниса--Море)
  - alpha_k = sin(angle(d_k, range(Pi_{k-1})))        (геометрия)

Тестовые задачи: Discrete BVP, Broyden Banded (n=50,100), x* находим
прогоном того же solver'а с очень жёстким tol -> x_ref.

Output:
  fig_pre_asymptotic_dm.pdf   --- ||E_k d_k|| / ||d_k|| vs k (DM-quantity)
  fig_pre_asymptotic_nu.pdf   --- nu_k vs k
  fig_pre_asymptotic_alpha.pdf --- alpha_k vs k для p=0 и p=5
  fig_pre_asymptotic_kpre.pdf  --- K_pre(delta) для разных delta
  pre_asymptotic.npz          --- сырые данные
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import cond, norm, solve, qr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------- задачи (импорт из существующего diag_sp_broyden) -----------


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


def discrete_bvp_J(x):
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


def discrete_bvp_x0(n):
    h = 1.0 / (n + 1)
    return 0.1 * np.array([i * h * (i * h - 1.0) for i in range(1, n + 1)])


def broyden_banded_F(x):
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        ji = [j for j in range(max(0, i - 5), min(n, i + 2)) if j != i]
        r[i] = x[i] * (2.0 + 5.0 * x[i] ** 2) + 1.0 - sum(
            x[j] * (1.0 + x[j]) for j in ji
        )
    return r


def broyden_banded_J(x):
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2.0 + 15.0 * x[i] ** 2
        for j in range(max(0, i - 5), min(n, i + 2)):
            if j == i:
                continue
            J[i, j] = -(1.0 + 2.0 * x[j])
    return J


def broyden_banded_x0(n):
    return -0.1 * np.ones(n)


# --------------- SP-Broyden c полным трекингом -----------------------


def sp_broyden_track(
    F, Jfun, x0, p_max, x_star=None,
    cond_thresh=1e3, tol=1e-13, maxiter=600,
):
    """SP-Broyden c расширенным трекингом для пре-асимптотики.

    Возвращает: список словарей, по одному на итерацию k.
    Поля: x, F, B, J(x), d, s, y, dist (||x-x*||),
          dm (||E_k d_k||/||d_k||), alpha_prev (sin∠(d_k, range(Pi_{k-1}))),
          p_eff, cond_Sp, range_Pi_prev (для следующей итерации).
    """
    n = len(x0)
    x = x0.copy().astype(float)
    Fx = F(x)
    B = np.eye(n)
    S_hist, Y_hist = [], []
    range_Pi_prev = None  # range of Pi_k от ПРЕДЫДУЩЕЙ итерации

    log = []
    log.append({
        "k": 0, "x": x.copy(), "Fnorm": float(norm(Fx)),
        "dist": float(norm(x - x_star)) if x_star is not None else float("nan"),
        "dm": float("nan"), "alpha_prev": float("nan"),
        "nu": float("nan"), "p_eff": 0,
    })

    for k in range(maxiter):
        if norm(Fx) < tol:
            break
        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break

        # ----- DM-quantity ||E_k d_k|| / ||d_k|| ДО шага -----
        Jx = Jfun(x)
        E = B - Jx
        dnorm = norm(d)
        dm = float(norm(E @ d) / dnorm) if dnorm > 0 else float("nan")

        # ----- alpha_k = sin∠(d_k, range(Pi_{k-1})) -----
        if range_Pi_prev is None:
            alpha_prev = float("nan")
        else:
            Q = range_Pi_prev  # ортонормальные столбцы range(Pi_{k-1})
            d_proj = Q @ (Q.T @ d)
            d_perp = d - d_proj
            alpha_prev = float(norm(d_perp) / dnorm) if dnorm > 0 else float("nan")

        # ----- шаг -----
        x_new = x + d
        Fx_new = F(x_new)
        if not np.all(np.isfinite(Fx_new)):
            break
        y = Fx_new - Fx
        s = d.copy()
        S_hist.append(s.copy())
        Y_hist.append(y.copy())

        # nu_k
        if x_star is not None:
            dist_new = float(norm(x_new - x_star))
            prev_dist = log[-1]["dist"]
            nu = dist_new / prev_dist if prev_dist > 0 and not np.isnan(prev_dist) else float("nan")
        else:
            dist_new = float("nan")
            nu = float("nan")

        # ----- адаптивное p -----
        Bs = B @ s
        p_eff = 0
        cond_used = 1.0
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                cG = cond(G)
                if cG < cond_thresh:
                    p_eff = p_try
                    cond_used = float(cG)
                else:
                    break
        if p_eff == 0:
            v = s
            range_Pi = s.reshape(-1, 1) / norm(s)
        else:
            cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols)
            G = Sp.T @ Sp
            e1 = np.zeros(p_eff + 1)
            e1[0] = 1.0
            try:
                v = Sp @ solve(G, e1)
                Q_pi, _ = qr(Sp)
                range_Pi = Q_pi
            except np.linalg.LinAlgError:
                v = s
                p_eff = 0
                range_Pi = s.reshape(-1, 1) / norm(s)

        denom = float(v @ s)
        if abs(denom) < 1e-12:
            v = s
            denom = float(s @ s)
            range_Pi = s.reshape(-1, 1) / norm(s)

        B = B + np.outer(y - Bs, v) / denom

        x = x_new
        Fx = Fx_new
        log.append({
            "k": k + 1, "x": x.copy(), "Fnorm": float(norm(Fx)),
            "dist": dist_new, "dm": dm, "alpha_prev": alpha_prev,
            "nu": nu, "p_eff": int(p_eff),
        })
        range_Pi_prev = range_Pi

        max_hist = max(p_max + 5, 25)
        if len(S_hist) > max_hist:
            S_hist.pop(0)
            Y_hist.pop(0)

    return log


def find_x_star(F, x0, n):
    """Находим x* через высокоточный SP-Broyden с большим p."""
    log = sp_broyden_track(
        F, lambda x: np.zeros((n, n)), x0, p_max=20,
        cond_thresh=1e8, tol=1e-15, maxiter=2000,
    )
    # Возьмём последнюю точку (с самой малой ||F||)
    x_star = log[-1]["x"]
    return x_star


def compute_K_pre(log, delta_star):
    """K_pre(delta*) = min{k : dm_j <= delta* для всех j >= k}.

    Возвращает k или len(log) если порог не достигнут.
    """
    dms = [entry["dm"] for entry in log if not np.isnan(entry["dm"])]
    if not dms:
        return len(log)
    K = len(dms)
    # ищем последнее k, где dm_k > delta*; K_pre это k+1
    last_violation = -1
    for i, dm in enumerate(dms):
        if dm > delta_star:
            last_violation = i
    return last_violation + 1


# ----------------- main -----------------


def main():
    OUTDIR = os.path.join(os.path.dirname(__file__), "mipt_thesis_master")
    PROBLEMS = [
        ("Discrete BVP", discrete_bvp_F, discrete_bvp_J, discrete_bvp_x0, 100),
        ("Broyden Banded", broyden_banded_F, broyden_banded_J, broyden_banded_x0, 100),
    ]
    PS = [0, 1, 2, 5, 10]
    DELTAS = [0.5, 0.3, 0.1, 0.05, 0.01]

    all_logs = {}  # (problem_name, p) -> log

    for pname, F, Jfun, x0fun, n in PROBLEMS:
        x0 = x0fun(n)
        # Найдём x*: запустим жёсткий solver один раз
        print(f"\n=== {pname} (n={n}) ===")
        # Используем p=10 для нахождения x*
        ref_log = sp_broyden_track(
            F, Jfun, x0, p_max=10, cond_thresh=1e3, tol=1e-13, maxiter=2000,
        )
        x_star = ref_log[-1]["x"]
        Fnorm_star = ref_log[-1]["Fnorm"]
        print(f"  x* found: ||F(x*)|| = {Fnorm_star:.2e}")

        for p in PS:
            log = sp_broyden_track(
                F, Jfun, x0, p_max=p, x_star=x_star,
                cond_thresh=1e3, tol=1e-12, maxiter=600,
            )
            all_logs[(pname, p)] = log
            K_pres = {d: compute_K_pre(log, d) for d in DELTAS}
            converged = log[-1]["Fnorm"] < 1e-10
            print(
                f"  p={p:2d}: total iter = {len(log)-1}, converged: {converged}, "
                f"K_pre(0.1) = {K_pres[0.1]}"
            )

    # ============ Фигуры ============
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(PS)))

    # Fig 1: DM-quantity vs k
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(12, 4))
    for ax, (pname, *_) in zip(axes, PROBLEMS):
        for c, p in zip(cmap, PS):
            log = all_logs[(pname, p)]
            ks = [e["k"] for e in log[1:]]
            dms = [e["dm"] for e in log[1:]]
            ax.semilogy(ks, dms, color=c, label=f"$p={p}$")
        ax.axhline(0.1, color="k", linestyle="--", alpha=0.5,
                   label=r"$\delta^*=0.1$")
        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(r"$\|E_k d_k\| / \|d_k\|$")
        ax.set_title(f"{pname}: Денниса--Море")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_pre_asymptotic_dm.pdf"))
    plt.close(fig)

    # Fig 2: nu_k vs k
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(12, 4))
    for ax, (pname, *_) in zip(axes, PROBLEMS):
        for c, p in zip(cmap, PS):
            log = all_logs[(pname, p)]
            ks = [e["k"] for e in log[2:]]
            nus = [e["nu"] for e in log[2:]]
            ax.plot(ks, nus, color=c, label=f"$p={p}$")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(r"$\nu_k = \|x_{k+1}-x^*\|/\|x_k-x^*\|$")
        ax.set_title(f"{pname}: Q-линейный коэффициент")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_pre_asymptotic_nu.pdf"))
    plt.close(fig)

    # Fig 3: alpha_prev vs k (только p=0 и p=5)
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(12, 4))
    for ax, (pname, *_) in zip(axes, PROBLEMS):
        log0 = all_logs[(pname, 0)]
        log5 = all_logs[(pname, 5)]
        ks0 = [e["k"] for e in log0[2:]]
        a0 = [e["alpha_prev"] for e in log0[2:]]
        ks5 = [e["k"] for e in log5[2:]]
        a5 = [e["alpha_prev"] for e in log5[2:]]
        ax.plot(ks0, a0, color="tab:red", label=r"$\alpha_k^{Br}$ (p=0)")
        ax.plot(ks5, a5, color="tab:blue", label=r"$\alpha_k^{SP}$ (p=5)")
        ax.set_xlabel("итерация $k$")
        ax.set_ylabel(r"$\sin\angle(d_k, \mathrm{range}(\Pi_{k-1}))$")
        ax.set_title(f"{pname}: угловой дефект")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_pre_asymptotic_alpha.pdf"))
    plt.close(fig)

    # Fig 4: K_pre(delta) vs p
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(12, 4))
    for ax, (pname, *_) in zip(axes, PROBLEMS):
        for c, d in zip(plt.cm.plasma(np.linspace(0, 0.9, len(DELTAS))), DELTAS):
            K_pres = []
            for p in PS:
                log = all_logs[(pname, p)]
                K_pres.append(compute_K_pre(log, d))
            ax.plot(PS, K_pres, "o-", color=c, label=fr"$\delta^*={d}$")
        ax.set_xlabel("$p$")
        ax.set_ylabel(r"$K_{\rm pre}(\delta^*)$")
        ax.set_title(f"{pname}: длина пре-асимптотики vs $p$")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PS)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_pre_asymptotic_kpre.pdf"))
    plt.close(fig)

    # Сохраняем ключевые данные
    K_pre_table = {}
    for (pname, p), log in all_logs.items():
        for d in DELTAS:
            K_pre_table[(pname, p, d)] = compute_K_pre(log, d)

    np.savez(
        os.path.join(os.path.dirname(__file__), "pre_asymptotic.npz"),
        problems=[p[0] for p in PROBLEMS],
        ps=np.array(PS),
        deltas=np.array(DELTAS),
    )
    print(f"\nДанные --> pre_asymptotic.npz")
    print(f"Фигуры --> {OUTDIR}/fig_pre_asymptotic_{{dm,nu,alpha,kpre}}.pdf")

    print("\n=== Сводная таблица K_pre(delta=0.1) ===")
    print(f"{'Задача':25s} | " + " | ".join(f"p={p:2d}" for p in PS))
    print("-" * 70)
    for pname, *_ in PROBLEMS:
        row = [str(K_pre_table[(pname, p, 0.1)]).rjust(4) for p in PS]
        print(f"{pname:25s} | " + " | ".join(row))


if __name__ == "__main__":
    main()
