"""
diag_global_study.py — диагностический пакет к §2.3.

Цель: эмпирически проверить, в какой мере bounded deterioration в PB играет
роль «эффективной глобализации», и где она перестаёт хватать.

Четыре эксперимента:

EXP-A  A/B по глобализации на одной задаче (Discrete BVP, n=100).
       p_max ∈ {0, 1, 5} × globalize ∈ {False, True}.
       20 стартов (как в Рис. spb_jacerr). Метрика: median ||F||, IQR,
       доля сошедшихся (||F|| ≤ 1e-10 за 600 итер).

EXP-B  История α_k в режиме (II) (Broyden Banded, n=10^4).
       L-PB, m=20, p_max=5, globalize=True. 4 старта. Гистограмма α_k
       и фракция α_k=1 (т.е. backtracking не сработал).

EXP-C  Диагностика несошедшихся стартов в (I) (Discrete BVP, p_max=1,
       globalize=False). Классификация хвоста ||F||: плато / осцилляция /
       расходимость.

EXP-D  Дальние x_0. Discrete BVP n=100, Δ ∈ {0.05, 0.2, 0.5, 1.0},
       p_max=5, globalize ∈ {False, True}. Метрика: доля сошедшихся.

Реализация PB-Direct-B следует diag_jacerr_stat.py точно (фиксированное
p = min(p_max, k), без адаптивного κ-клиппинга), плюс опциональный
backtracking-Армихо по ψ = ½‖F‖² (стандартное условие c1=1e-4, max_back=25).

Выходы:
  mipt_thesis_master/fig_global_AB.pdf        — EXP-A: 2×3 grid traces.
  mipt_thesis_master/fig_global_alpha.pdf     — EXP-B: гистограмма α_k.
  mipt_thesis_master/fig_global_perturb.pdf   — EXP-D: bar-chart по Δ.
  global_study_report.txt                     — текстовый отчёт.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
from numpy.linalg import norm, solve, cond
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diag_highdim as dh   # F-функции (broyden_banded_F и т.п.)

SEED = 20260502   # тот же, что в diag_jacerr_stat.py — даёт ансамбль из Рис. spb_jacerr
TOL = 1e-10
MAXIT_BVP = 600
N_STARTS = 20
N_BVP = 100

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mipt_thesis_master")
REPORT = []  # накапливаем сообщения


def _log(msg: str = ""):
    print(msg, flush=True)
    REPORT.append(msg)


# ---------------------------------------------------------------------------
# Discrete BVP (MGH #28), n = 100, аналитический Якобиан.
# Скопировано из diag_jacerr_stat.py для гарантии того же solver-контракта.
# ---------------------------------------------------------------------------

def bvp_F(x: np.ndarray) -> np.ndarray:
    n = x.size
    h = 1.0 / (n + 1)
    t = h * np.arange(1, n + 1)
    F = np.empty_like(x)
    F[0]  = 2 * x[0] - x[1]                    + 0.5 * h * h * (x[0]  + t[0]  + 1.0) ** 3
    F[-1] = 2 * x[-1] - x[-2]                  + 0.5 * h * h * (x[-1] + t[-1] + 1.0) ** 3
    F[1:-1] = 2 * x[1:-1] - x[:-2] - x[2:]     + 0.5 * h * h * (x[1:-1] + t[1:-1] + 1.0) ** 3
    return F


def bvp_J(x: np.ndarray) -> np.ndarray:
    n = x.size
    h = 1.0 / (n + 1)
    t = h * np.arange(1, n + 1)
    J = 2.0 * np.eye(n) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    J[np.arange(n), np.arange(n)] += 1.5 * h * h * (x + t + 1.0) ** 2
    return J


def bvp_x0_default(n: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    t = h * np.arange(1, n + 1)
    return 0.1 * t * (t - 1.0)


# ---------------------------------------------------------------------------
# Powell singular function (MGH #13). n кратно 4.
# F: R^n -> R^n, x* = 0, J(x*) = 0 (singular!) — классический хард-тест.
# ---------------------------------------------------------------------------

def powell_singular_F(x: np.ndarray) -> np.ndarray:
    n = x.size
    assert n % 4 == 0, "Powell singular: n must be a multiple of 4"
    F = np.empty_like(x)
    x1 = x[0::4]; x2 = x[1::4]; x3 = x[2::4]; x4 = x[3::4]
    F[0::4] = x1 + 10.0 * x2
    F[1::4] = np.sqrt(5.0) * (x3 - x4)
    F[2::4] = (x2 - 2.0 * x3) ** 2
    F[3::4] = np.sqrt(10.0) * (x1 - x4) ** 2
    return F


def powell_singular_x0(n: int) -> np.ndarray:
    """Стандартный старт MGH: каждый блок (3, -1, 0, 1)."""
    x0 = np.empty(n)
    x0[0::4] = 3.0
    x0[1::4] = -1.0
    x0[2::4] = 0.0
    x0[3::4] = 1.0
    return x0


# ---------------------------------------------------------------------------
# PB direct-B (fixed p = min(p_max, k)) с опциональным Армихо по ψ=½‖F‖².
# ---------------------------------------------------------------------------

def armijo_step(F, x, Fx, d, c1=1e-4, max_back=25):
    """Backtracking-Армихо по ψ=½‖F‖²: ψ_new ≤ ψ_old (1 - 2 c1 α)."""
    psi0 = 0.5 * float(Fx @ Fx)
    alpha = 1.0
    for _ in range(max_back):
        x_new = x + alpha * d
        Fx_new = F(x_new)
        if not np.all(np.isfinite(Fx_new)):
            alpha *= 0.5
            continue
        psi_new = 0.5 * float(Fx_new @ Fx_new)
        if psi_new <= psi0 * (1.0 - 2.0 * c1 * alpha):
            return alpha, x_new, Fx_new
        alpha *= 0.5
    return alpha, x_new, Fx_new


def pb_solve(F, x0, p_max=0, maxit=MAXIT_BVP, tol=TOL, globalize=False):
    """Projected Broyden direct-B; зеркало diag_jacerr_stat.py с опц. Армихо.
    Возвращает dict с историей ‖F‖, α_k, флагом converged, числом итераций.
    """
    n = len(x0)
    x = x0.astype(float).copy()
    Fx = F(x)
    B = np.eye(n)
    S_hist = []
    res = [float(norm(Fx))]
    alphas = []
    converged = False
    for k in range(maxit):
        if res[-1] < tol:
            converged = True
            break
        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break
        if globalize:
            alpha, x_new, Fx_new = armijo_step(F, x, Fx, d)
        else:
            x_new = x + d
            Fx_new = F(x_new)
            alpha = 1.0
        if not np.all(np.isfinite(Fx_new)):
            break
        s = x_new - x
        y = Fx_new - Fx
        S_hist.append(s.copy())
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
        res.append(float(norm(Fx)))
        alphas.append(alpha)
        if len(S_hist) > p_max + 5:
            S_hist.pop(0)
    return dict(res=np.array(res), alphas=np.array(alphas),
                converged=converged, iters=len(res) - 1, x_final=x)


# ---------------------------------------------------------------------------
# Сборщик статистики по ансамблю стартов.
# ---------------------------------------------------------------------------

def run_ensemble(F, starts, **solver_kwargs):
    """Запустить pb_solve на ансамбле и собрать траектории + статистику."""
    out = []
    for x0 in starts:
        res = pb_solve(F, x0, **solver_kwargs)
        out.append(res)
    return out


def median_iqr_curve(results, key="res", maxlen=None):
    """Медиана/Q1/Q3 по итерациям, обрезанные на длине, где ≤ половина живая."""
    arrs = [r[key] for r in results]
    if maxlen is None:
        maxlen = max(len(a) for a in arrs)
    M = np.full((len(arrs), maxlen), np.nan)
    for i, a in enumerate(arrs):
        M[i, :len(a)] = a
    alive = np.sum(~np.isnan(M), axis=0)
    cut = maxlen
    for k in range(maxlen):
        if alive[k] < (len(arrs) + 1) // 2:
            cut = k
            break
    med = np.nanmedian(M[:, :cut], axis=0)
    q1 = np.nanpercentile(M[:, :cut], 25, axis=0)
    q3 = np.nanpercentile(M[:, :cut], 75, axis=0)
    return np.arange(cut), med, q1, q3


# ===========================================================================
# EXP-A. A/B по глобализации (Discrete BVP, n=100, 20 стартов).
# ===========================================================================

def exp_A():
    _log("\n" + "=" * 72)
    _log("EXP-A. A/B по глобализации, Discrete BVP n=100, 20 стартов.")
    _log("=" * 72)

    rng = np.random.default_rng(SEED)
    x0_def = bvp_x0_default(N_BVP)
    starts = []
    for _ in range(N_STARTS):
        u = rng.standard_normal(N_BVP); u /= norm(u)
        starts.append(x0_def + 0.05 * u)

    p_list = [0, 1, 5]
    glob_list = [False, True]
    results = {}    # (p, glob) -> list of dicts
    F = bvp_F
    t0 = time.time()
    for p in p_list:
        for glob in glob_list:
            res = run_ensemble(F, starts, p_max=p, globalize=glob, maxit=MAXIT_BVP)
            results[(p, glob)] = res
            conv = sum(r["converged"] for r in res)
            med_iters = int(np.median([r["iters"] for r in res if r["converged"]])) \
                if conv > 0 else -1
            _log(f"  p_max={p}, globalize={glob}: conv {conv}/{N_STARTS}, "
                 f"median iters (conv) = {med_iters}")
    _log(f"  wall-time EXP-A: {time.time() - t0:.1f}s")

    # Plot 2×3.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for i, glob in enumerate(glob_list):
        for j, p in enumerate(p_list):
            ax = axes[i, j]
            xs, med, q1, q3 = median_iqr_curve(results[(p, glob)])
            ax.semilogy(xs, med, lw=1.6, color="C0")
            ax.fill_between(xs, q1, q3, color="C0", alpha=0.18, lw=0)
            ax.axhline(TOL, ls=":", color="grey", lw=0.6)
            ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
            ax.set_title(f"p_max={p}, "
                         f"{'+Армихо' if glob else 'полн. шаг'}",
                         fontsize=10)
            if j == 0:
                ax.set_ylabel(r"$\|F(x_k)\|_2$")
            if i == 1:
                ax.set_xlabel("итерация $k$")
    fig.suptitle("EXP-A: Discrete BVP $n=100$, 20 случайных стартов "
                 "($\\Delta=0.05$); медиана $\\pm$ IQR", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_global_AB.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _log(f"  saved: {out}")
    return results


# ===========================================================================
# EXP-B. Гистограмма α_k в режиме (II).
# Делаем на Discrete BVP n=100 (α_k собрана в EXP-A) и отдельно на
# Broyden Banded n=10^4 (через diag_highdim.lsp_broyden, монкей-патч).
# ===========================================================================

def exp_B_bvp(results_A):
    _log("\n" + "=" * 72)
    _log("EXP-B.1. Гистограмма α_k для Discrete BVP, p_max ∈ {0,1,5}, +Армихо.")
    _log("=" * 72)
    alphas_by_p = {}
    for p in [0, 1, 5]:
        all_alphas = np.concatenate([r["alphas"] for r in results_A[(p, True)]])
        alphas_by_p[p] = all_alphas
        frac_full = float(np.mean(all_alphas >= 1.0 - 1e-12))
        frac_half_or_less = float(np.mean(all_alphas <= 0.5))
        _log(f"  p_max={p}: total α-values = {len(all_alphas)}, "
             f"frac(α=1.0) = {frac_full:.3f}, "
             f"frac(α≤0.5) = {frac_half_or_less:.3f}, "
             f"min α = {all_alphas.min():.2e}")
    return alphas_by_p


def lsp_broyden_with_alpha(F, x0, m=20, p_max=5, maxiter=400, tol=TOL):
    """Аналог diag_highdim.lsp_broyden, но фиксирует α_k и не клиппит p.

    Реализация — точная копия dh.lsp_broyden с globalize=True (Армихо c1=1e-4),
    с добавкой записи α_k. Используется для EXP-B.2.
    """
    from diag_highdim import apply_H_pairs, step_cap, armijo_step
    n = x0.size
    x = x0.astype(float).copy()
    Fx = F(x)
    pairs = []
    res = [float(norm(Fx))]
    alphas = []
    converged = False
    for k in range(maxiter):
        if res[-1] < tol:
            converged = True
            break
        Hg, _, _ = apply_H_pairs(Fx, pairs)
        d = -Hg
        if not np.all(np.isfinite(d)):
            break
        d = step_cap(d, x)
        alpha, x_new, Fx_new, _ = armijo_step(F, x, Fx, d)
        if not np.all(np.isfinite(Fx_new)):
            break
        s = x_new - x
        y = Fx_new - Fx
        # v_k: фиксированный p = min(p_max, |pairs|)
        s_full = [pp[0] for pp in pairs] + [s]
        p_eff = min(p_max, len(s_full) - 1)
        if p_eff == 0:
            v = s
        else:
            cols = [s_full[-1 - j] for j in range(p_eff + 1)]
            Sp = np.column_stack(cols); G = Sp.T @ Sp
            e1 = np.zeros(p_eff + 1); e1[0] = 1.0
            try:
                v = Sp @ solve(G, e1)
            except np.linalg.LinAlgError:
                v = s
        Hy_test, _, _ = apply_H_pairs(y, pairs)
        denom = float(v @ Hy_test)
        if abs(denom) < 1e-14 * (norm(v) * norm(Hy_test) + 1e-30):
            x, Fx = x_new, Fx_new
        else:
            pairs.append((s.copy(), y.copy(), v.copy()))
            x, Fx = x_new, Fx_new
        res.append(float(norm(Fx)))
        alphas.append(alpha)
        while len(pairs) > m:
            pairs.pop(0)
    return dict(res=np.array(res), alphas=np.array(alphas),
                converged=converged, iters=len(res) - 1)


def exp_B_banded():
    _log("\n" + "=" * 72)
    _log("EXP-B.2. Гистограмма α_k для Broyden Banded, n=10^4, L-PB(m=20,p_max=5).")
    _log("=" * 72)
    n = 10_000
    rng = np.random.default_rng(SEED + 1)
    x0_def = dh.broyden_banded_x0(n)
    F = dh.broyden_banded_F
    starts = []
    for _ in range(4):
        u = rng.standard_normal(n); u /= norm(u)
        starts.append(x0_def + 0.05 * u)
    all_alphas = []
    for i, x0 in enumerate(starts):
        t0 = time.time()
        r = lsp_broyden_with_alpha(F, x0, m=20, p_max=5, maxiter=200, tol=TOL)
        _log(f"  start {i}: iters={r['iters']}, conv={r['converged']}, "
             f"min α={r['alphas'].min():.3f}, frac(α=1)={np.mean(r['alphas']>=1-1e-12):.3f}, "
             f"wall {time.time()-t0:.1f}s")
        all_alphas.append(r["alphas"])
    all_alphas = np.concatenate(all_alphas)
    frac_full = float(np.mean(all_alphas >= 1.0 - 1e-12))
    _log(f"  TOTAL: {len(all_alphas)} α-values, frac(α=1.0) = {frac_full:.3f}, "
         f"frac(α≤0.5)={float(np.mean(all_alphas<=0.5)):.3f}, "
         f"min α = {all_alphas.min():.2e}")
    return all_alphas


def plot_alphas(alphas_bvp_by_p, alphas_banded):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bins = np.linspace(0, 1.05, 30)
    for p in [0, 1, 5]:
        ax.hist(alphas_bvp_by_p[p], bins=bins, alpha=0.55, label=f"p_max={p}")
    ax.set_xlabel(r"$\alpha_k$")
    ax.set_ylabel("count")
    ax.set_title("Discrete BVP, n=100, 20 стартов, +Армихо")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    ax.hist(alphas_banded, bins=bins, color="C2", alpha=0.7)
    ax.set_xlabel(r"$\alpha_k$")
    ax.set_ylabel("count")
    ax.set_title("Broyden Banded, n=10^4, L-PB(m=20,p=5), +Армихо")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle("EXP-B: распределение длины шага Армихо $\\alpha_k$", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_global_alpha.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _log(f"  saved: {out}")


# ===========================================================================
# EXP-C. Несходящиеся старты в (I): p_max=1, без Армихо. Классификация хвоста.
# ===========================================================================

def classify_tail(res, tail_window=30):
    """Классификация хвоста ‖F‖: plateau / oscillation / divergence."""
    if len(res) < tail_window + 1:
        return "short"
    tail = res[-tail_window:]
    log_tail = np.log10(tail + 1e-300)
    # медиана-ослабления
    drift = log_tail[-1] - log_tail[0]
    std = float(np.std(log_tail))
    if drift > 0.5:
        return "divergence"
    if std < 0.05 and abs(drift) < 0.1:
        return "plateau"
    return "oscillation"


def exp_C(results_A):
    _log("\n" + "=" * 72)
    _log("EXP-C. Диагностика несошедшихся стартов: Discrete BVP, p_max=1, без Армихо.")
    _log("=" * 72)
    res_list = results_A[(1, False)]
    n_starts = len(res_list)
    bins = {"plateau": [], "oscillation": [], "divergence": [], "short": []}
    for i, r in enumerate(res_list):
        if r["converged"]:
            continue
        kind = classify_tail(r["res"])
        bins[kind].append(i)
        _log(f"  start {i}: iters={r['iters']}, last ‖F‖={r['res'][-1]:.3e}, "
             f"min ‖F‖={r['res'].min():.3e}, tail_class={kind}")
    n_conv = sum(r["converged"] for r in res_list)
    _log(f"  ИТОГО: conv {n_conv}/{n_starts}, "
         f"plateau {len(bins['plateau'])}, "
         f"oscillation {len(bins['oscillation'])}, "
         f"divergence {len(bins['divergence'])}, "
         f"short {len(bins['short'])}")
    return bins


# ===========================================================================
# EXP-D. Дальние x_0. Discrete BVP n=100, Δ ∈ {0.05, 0.2, 0.5, 1.0}.
# ===========================================================================

def exp_D():
    _log("\n" + "=" * 72)
    _log("EXP-D. Влияние Δ возмущения x_0, Discrete BVP n=100, p_max=5.")
    _log("=" * 72)
    rng = np.random.default_rng(SEED + 2)
    x0_def = bvp_x0_default(N_BVP)
    deltas = [0.05, 0.2, 0.5, 1.0]
    glob_list = [False, True]
    conv_table = {}
    for delta in deltas:
        # перестроим ансамбль с этим Δ
        starts = []
        for _ in range(N_STARTS):
            u = rng.standard_normal(N_BVP); u /= norm(u)
            starts.append(x0_def + delta * u)
        for glob in glob_list:
            res = run_ensemble(bvp_F, starts, p_max=5, globalize=glob, maxit=MAXIT_BVP)
            conv = sum(r["converged"] for r in res)
            med_iters = int(np.median([r["iters"] for r in res if r["converged"]])) \
                if conv > 0 else -1
            conv_table[(delta, glob)] = (conv, med_iters)
            _log(f"  Δ={delta:>4}, glob={glob}: conv {conv}/{N_STARTS}, "
                 f"med_iters_conv={med_iters}")

    # bar-chart: для каждого Δ две полосы (без / с Армихо)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.36
    xs = np.arange(len(deltas))
    bar_no = [conv_table[(d, False)][0] for d in deltas]
    bar_yes = [conv_table[(d, True)][0] for d in deltas]
    ax.bar(xs - width / 2, bar_no, width, label="полн. шаг", color="C3", alpha=0.8)
    ax.bar(xs + width / 2, bar_yes, width, label="+Армихо", color="C0", alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Δ={d}" for d in deltas])
    ax.set_ylabel(f"сошлось из {N_STARTS}")
    ax.set_title("EXP-D: устойчивость PB(p=5) к дальности старта от $x_0^{\\rm def}$")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5, axis="y")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_global_perturb.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _log(f"  saved: {out}")
    return conv_table


# ===========================================================================
# EXP-E. Powell singular (MGH #13): хард-задача с сингулярным якобианом в x*.
#        Сравнение PB(p=5) и Бройдена (p=0), без/с Армихо, на 20 стартах.
# ===========================================================================

def exp_E():
    _log("\n" + "=" * 72)
    _log("EXP-E. Powell singular, n=20, p_max ∈ {0,5}, без/с Армихо.")
    _log("=" * 72)
    n = 20
    rng = np.random.default_rng(SEED + 7)
    x0_def = powell_singular_x0(n)
    F = powell_singular_F
    starts = []
    for _ in range(N_STARTS):
        u = rng.standard_normal(n); u /= norm(u)
        starts.append(x0_def + 1.0 * u)   # Δ=1.0: значимое возмущение
    p_list = [0, 5]
    glob_list = [False, True]
    table = {}
    for p in p_list:
        for glob in glob_list:
            res = run_ensemble(F, starts, p_max=p, globalize=glob, maxit=1500)
            # Powell singular: tol по ‖F‖² ≤ 1e-10 (как и везде)
            conv = sum(r["converged"] for r in res)
            iters_conv = [r["iters"] for r in res if r["converged"]]
            med_iters = int(np.median(iters_conv)) if iters_conv else -1
            min_resid = float(np.median([r["res"].min() for r in res]))
            table[(p, glob)] = (conv, med_iters, min_resid)
            _log(f"  p_max={p}, glob={glob}: conv {conv}/{N_STARTS}, "
                 f"med_iters_conv={med_iters}, "
                 f"med(min ‖F‖) = {min_resid:.3e}")
    return table


# ===========================================================================
# main
# ===========================================================================

def main():
    t_start = time.time()
    _log(f"diag_global_study.py — old wall-clock start.")
    _log(f"seed = {SEED}, tol = {TOL:g}, n_BVP = {N_BVP}, "
         f"maxit_BVP = {MAXIT_BVP}, n_starts = {N_STARTS}")

    results_A = exp_A()
    alphas_bvp = exp_B_bvp(results_A)
    alphas_banded = exp_B_banded()
    plot_alphas(alphas_bvp, alphas_banded)
    bins_C = exp_C(results_A)
    table_D = exp_D()
    table_E = exp_E()

    _log("\n" + "=" * 72)
    _log(f"TOTAL wall-time: {time.time() - t_start:.1f}s")

    # сохраняем отчёт
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "global_study_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(REPORT))
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
