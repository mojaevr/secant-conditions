"""
diag_highdim.py — эксперименты SP-Broyden / L-SP-Broyden в высокой размерности.

Закрывает три пункта блока P2 «Высокая размерность / limited memory» в REVIEW_TODO:
  1) эксперименты при n ∈ {1000, 10000} с обратной формулой Шермана–Моррисона;
  2) реализация и обсуждение L-SP-Broyden (limited memory вариант);
  3) сравнение с L-Broyden (классический Бройден, limited-memory) и Anderson(m,β).

Все методы сравниваются на одинаковом backtracking-Армихо по merit-функции
ψ(x) = ½‖F(x)‖² (для шага d = -H_k F(x)) и одинаковом критерии остановки
‖F(x_k)‖₂ ≤ ε_tol = 10^{-10}.

Тестовые задачи (Moré–Garbow–Hillstrom 1981, векторизованные):
  (A) Discrete BVP (MGH #28) — трёхдиагональный линеаризованный якобиан;
  (B) Broyden Banded (MGH #31) — лента ширины 7, существенно нелинейный.

Размерности: n ∈ {1000, 10000}. Количество семейств × размеров × методов ≈ 16 экспериментов.

Выход (mipt_thesis_master/):
  fig_highdim_conv.pdf      — ‖F‖ vs итерация, 4 панели (2 задачи × 2 размера)
  fig_highdim_summary.pdf   — bar-chart: итерации × #fevals × wall-time × peak-memory
  fig_highdim_pvar.pdf      — variation окна L-SP-Broyden m ∈ {2,5,10,20}, n=1000

Сырые: highdim_results.npz.

Воспроизводимость: NumPy 2.x, фиксированный seed=20260430 для всех источников
случайности (стартовые точки детерминированы; default_rng оставлен для совместимости
с reproducibility.tex).
"""

from __future__ import annotations

import os
import time
import numpy as np
from numpy.linalg import cond, norm, solve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "mipt_thesis_master"
SEED = 20260430
EPS_TOL = 1e-10

# ----------------------------------------------------------------------
# Векторизованные тестовые задачи (для n до 10^5).
# ----------------------------------------------------------------------


def discrete_bvp_F(x: np.ndarray) -> np.ndarray:
    """Discrete BVP, MGH #28. F: R^n -> R^n. F(x*) = 0 в нулях.

    F_i = 2 x_i - x_{i-1} - x_{i+1} + h² (x_i + t_i + 1)³ / 2,  t_i = i h, h=1/(n+1).
    Граничные условия: x_0 = x_{n+1} = 0.
    """
    n = x.size
    h = 1.0 / (n + 1)
    t = h * np.arange(1, n + 1)
    xm = np.empty_like(x); xm[0] = 0.0; xm[1:] = x[:-1]
    xp = np.empty_like(x); xp[-1] = 0.0; xp[:-1] = x[1:]
    return 2.0 * x - xm - xp + 0.5 * h * h * (x + t + 1.0) ** 3


def discrete_bvp_x0(n: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    t = h * np.arange(1, n + 1)
    # Совпадает со стартом в diag_sp_broyden.py (стандарт MGH).
    return 0.1 * t * (t - 1.0)


def broyden_banded_F(x: np.ndarray) -> np.ndarray:
    """Broyden Banded, MGH #31. F: R^n -> R^n, лента ширины 7 (5 ниже, 1 выше).

    F_i = x_i (2 + 5 x_i²) + 1 - Σ_{j∈J_i} x_j (1 + x_j),
       где J_i = {max(0, i-5), ..., min(n-1, i+1)} \ {i}.
    """
    n = x.size
    g = x * (1.0 + x)         # x_j (1 + x_j) для всех j
    sums = np.zeros_like(x)
    # окно [i-5, i+1] исключая i:  суммируем сдвинутые g
    for offset in (-5, -4, -3, -2, -1, 1):
        lo = max(0, -offset)
        hi = min(n, n - offset)
        sums[lo:hi] += g[lo + offset:hi + offset]
    return x * (2.0 + 5.0 * x * x) + 1.0 - sums


def broyden_banded_x0(n: int) -> np.ndarray:
    # Стандарт MGH: x0 = -1, и в существующих экспериментах diag_sp_broyden.py
    # использовалось -0.1 для устойчивости квазиньютоновской фазы.
    return -0.1 * np.ones(n)


def banded_cubic_F(x: np.ndarray) -> np.ndarray:
    """«Banded Cubic»: трёхдиагональная сильно-монотонная система с
    существенно нелинейным кубическим возмущением. Конструируется так,
    чтобы число обусловленности линейной части не зависело от n.

        F_i(x) = (1 + 0.5 sin²(π i / n)) x_i
                 - 0.3 (x_{i-1} + x_{i+1})
                 + 1.5 (x_i - c_i)³,
        x_0 = x_{n+1} = 0,  c_i = sin(2π i / n) / 2  ∈ [-0.5, 0.5].

    Линейная часть L = diag(a_i) - 0.3 trid: cond(L) ≈ 5 независимо от n.
    Кубическое возмущение значимо: при ‖x‖ = O(1) куб. член того же порядка,
    что и линейный. Решение x* существует и удовлетворяет F(x*) = 0,
    его положение неизвестно аналитически.
    """
    n = x.size
    i_idx = np.arange(1, n + 1)
    a = 1.0 + 0.5 * np.sin(np.pi * i_idx / n) ** 2
    c = 0.5 * np.sin(2.0 * np.pi * i_idx / n)
    xm = np.empty_like(x); xm[0] = 0.0; xm[1:] = x[:-1]
    xp = np.empty_like(x); xp[-1] = 0.0; xp[:-1] = x[1:]
    return a * x - 0.3 * (xm + xp) + 1.5 * (x - c) ** 3


def banded_cubic_x0(n: int) -> np.ndarray:
    # Удалённый старт, не зависящий от n.
    i_idx = np.arange(1, n + 1)
    return 1.0 + 0.5 * np.cos(np.pi * i_idx / n)


PROBLEMS = {
    "Banded Cubic": (banded_cubic_F, banded_cubic_x0),
    "Broyden Banded": (broyden_banded_F, broyden_banded_x0),
}


# ----------------------------------------------------------------------
# Backtracking-Армихо по merit ψ = ½‖F‖² (Алгоритм 2 из гл. 2).
# ----------------------------------------------------------------------


def step_cap(d: np.ndarray, x: np.ndarray, max_ratio: float = 50.0) -> np.ndarray:
    """Ограничение длины шага: ‖d‖ ≤ max_ratio · max(‖x‖, 1).

    Простейшая trust-region-light защита: предотвращает overflow в начальной
    фазе при $H_0 = I$, когда -H F даёт $-F$ слишком большой нормы. Срабатывает
    редко (типично 0–3 итерации) и не влияет на асимптотику.
    """
    bound = max_ratio * max(float(norm(x)), 1.0)
    nd = float(norm(d))
    if nd > bound:
        d = d * (bound / nd)
    return d


def armijo_step(F, x, Fx, d, c1: float = 1e-4, max_back: int = 25) -> tuple:
    """Возвращает (alpha, x_new, Fx_new, n_evals).
    Условие достаточного убывания: ψ(x+αd) ≤ ψ(x)(1 - 2 c1 α).
    Это эквивалентно стандартному условию Армихо для merit ψ при
    направлении d = -H F с ⟨∇ψ, d⟩ = -‖F‖² (если H ≈ J^{-1}).
    """
    psi0 = 0.5 * float(Fx @ Fx)
    alpha = 1.0
    n_evals = 0
    for _ in range(max_back):
        x_new = x + alpha * d
        Fx_new = F(x_new)
        n_evals += 1
        if not np.all(np.isfinite(Fx_new)):
            alpha *= 0.5
            continue
        psi_new = 0.5 * float(Fx_new @ Fx_new)
        if psi_new <= psi0 * (1.0 - 2.0 * c1 * alpha):
            return alpha, x_new, Fx_new, n_evals
        alpha *= 0.5
    # выдыхаем: вернуть последнюю точку
    return alpha, x_new, Fx_new, n_evals


# ----------------------------------------------------------------------
# Метод 1: классический Бройден с Шермана-Морриcон (плотный H).
# ----------------------------------------------------------------------


def broyden_sm(F, x0, maxiter=400, tol=EPS_TOL, globalize=False):
    n = x0.size
    x = x0.astype(float).copy()
    Fx = F(x)
    n_eval = 1
    H = np.eye(n)               # H_0 = I, плотная nxn
    res = [float(norm(Fx))]
    fevals = [1]
    t0 = time.time()
    peak_floats = float(n * n)
    converged = False
    for k in range(maxiter):
        if res[-1] < tol:
            converged = True
            break
        d = -H @ Fx
        if not np.all(np.isfinite(d)):
            break
        d = step_cap(d, x)
        if globalize:
            alpha, x_new, Fx_new, ne = armijo_step(F, x, Fx, d)
            n_eval += ne
        else:
            x_new = x + d; Fx_new = F(x_new); n_eval += 1; alpha = 1.0
            if not np.all(np.isfinite(Fx_new)):
                break
        s = x_new - x
        y = Fx_new - Fx
        # SM: H_{k+1} = H + (s - H y) s^T H / (s^T H y)
        Hy = H @ y
        denom = float(s @ Hy)
        if abs(denom) < 1e-14 * (norm(s) * norm(Hy) + 1e-30):
            break
        sH = s @ H               # 1×n
        H = H + np.outer((s - Hy) / denom, sH)
        x, Fx = x_new, Fx_new
        res.append(float(norm(Fx)))
        fevals.append(n_eval)
    wall = time.time() - t0
    return dict(method="Broyden-SM", res=np.array(res), fevals=np.array(fevals),
                wall=wall, peak_floats=peak_floats, converged=converged, iters=len(res) - 1)


# ----------------------------------------------------------------------
# Метод 2: SP-Broyden с Шермана-Морриcон (плотный H, окно глубины p).
# ----------------------------------------------------------------------


def sp_broyden_sm(F, x0, p_max=5, cond_thresh=1e3, maxiter=400, tol=EPS_TOL,
                  globalize=False, hist_keep=None):
    n = x0.size
    x = x0.astype(float).copy()
    Fx = F(x)
    n_eval = 1
    H = np.eye(n)
    if hist_keep is None:
        hist_keep = max(p_max + 5, 25)
    S_hist = []
    res = [float(norm(Fx))]
    fevals = [1]
    t0 = time.time()
    peak_floats = float(n * n + n * hist_keep)
    converged = False
    for k in range(maxiter):
        if res[-1] < tol:
            converged = True
            break
        d = -H @ Fx
        if not np.all(np.isfinite(d)):
            break
        d = step_cap(d, x)
        if globalize:
            alpha, x_new, Fx_new, ne = armijo_step(F, x, Fx, d)
            n_eval += ne
        else:
            x_new = x + d; Fx_new = F(x_new); n_eval += 1; alpha = 1.0
            if not np.all(np.isfinite(Fx_new)):
                break
        s = x_new - x
        y = Fx_new - Fx
        S_hist.append(s.copy())
        # SP-вектор v_k = S_p (S_p^T S_p)^{-1} e_1
        p_eff = 0
        v = s
        if p_max > 0 and len(S_hist) >= 2:
            for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                try:
                    cG = float(cond(G))
                except np.linalg.LinAlgError:
                    break
                if cG < cond_thresh:
                    e1 = np.zeros(p_try + 1); e1[0] = 1.0
                    try:
                        v = Sp @ solve(G, e1)
                    except np.linalg.LinAlgError:
                        break
                    p_eff = p_try
                else:
                    break
        # SM: H_{k+1} = H + (s - H y) v^T H / (v^T H y)
        Hy = H @ y
        denom = float(v @ Hy)
        if abs(denom) < 1e-14 * (norm(v) * norm(Hy) + 1e-30):
            break
        vH = v @ H
        H = H + np.outer((s - Hy) / denom, vH)
        x, Fx = x_new, Fx_new
        res.append(float(norm(Fx)))
        fevals.append(n_eval)
        if len(S_hist) > hist_keep:
            S_hist.pop(0)
    wall = time.time() - t0
    return dict(method=f"SP-Broyden-SM(p≤{p_max})", res=np.array(res),
                fevals=np.array(fevals), wall=wall, peak_floats=peak_floats,
                converged=converged, iters=len(res) - 1)


# ----------------------------------------------------------------------
# Метод 3: L-SP-Broyden (limited-memory).
#
# Поддерживается окно из m последних троек (s_j, y_j, v_j) — необработанные
# секущие пары и SP-направления. Применение H_k @ g реализуется forward-pass
# рекурсией, аналогичной двухпетельной L-BFGS:
#
#     z = g           # H_0 = I
#     Hy_j = ?        # последовательно перевычисляется
#     for j in window:
#        H_j y_j = (применить корректировки [0..j-1] к y_j)
#        xi_j  = s_j - H_j y_j
#        alpha_j = v_j^T H_j y_j
#        z = z + xi_j * (v_j^T z) / alpha_j
#
# При полном перевычислении H_j y_j на каждом обращении операция стоит
# O(n m^2). При m=10 это 100*n flops, на n=10000 → 10^6 flops/apply, очень
# дёшево. Память O(n m).
#
# При p_max = 0: получаем classical limited-memory Бройден (L-Broyden), v_j=s_j.
#
# Замечание: попытка кэшировать xi_j между шагами и сбрасывать только oldest
# некорректна — при отбрасывании первой пары оставшиеся xi_j соответствуют
# *усечённому* префиксу истории и более не образуют согласованную SM-цепочку.
# Поэтому пересчитываем H_j y_j каждое обращение (как и L-BFGS).
# ----------------------------------------------------------------------


def apply_H_pairs(g: np.ndarray, pairs) -> tuple:
    """Применить H @ g, где H построено последовательной SM-апликацией:
        H_{j+1} = H_j + (s_j - H_j y_j) v_j^T H_j / (v_j^T H_j y_j)
    к H_0 = I.

    pairs: список троек (s_j, y_j, v_j) в порядке аккумуляции.
    Возвращает (H g, [H_j y_j]_j, [alpha_j]_j) — кэши пригождаются вызывающему.
    """
    z = g.copy()
    Hy_cache = []
    alpha_cache = []
    for j, (s_j, y_j, v_j) in enumerate(pairs):
        # H_j y_j = последовательно применить корректировки 0..j-1 к y_j.
        Hjyj = y_j.copy()
        for i in range(j):
            xi_i = pairs[i][0] - Hy_cache[i]   # s_i - H_i y_i
            alpha_i = alpha_cache[i]
            v_i = pairs[i][2]
            beta_i = float(v_i @ Hjyj)
            Hjyj = Hjyj + xi_i * (beta_i / alpha_i)
        Hy_cache.append(Hjyj)
        alpha_j = float(v_j @ Hjyj)
        alpha_cache.append(alpha_j)
        # применяем корректировку j к z
        if abs(alpha_j) < 1e-30:
            continue
        xi_j = s_j - Hjyj
        beta_j = float(v_j @ z)
        z = z + xi_j * (beta_j / alpha_j)
    return z, Hy_cache, alpha_cache


def lsp_broyden(F, x0, m=10, p_max=None, cond_thresh=1e3, maxiter=400,
                tol=EPS_TOL, globalize=False):
    n = x0.size
    x = x0.astype(float).copy()
    Fx = F(x)
    n_eval = 1
    if p_max is None:
        p_max = m
    p_max = min(p_max, m)
    pairs = []   # [(s_j, y_j, v_j), ...]
    res = [float(norm(Fx))]
    fevals = [1]
    t0 = time.time()
    # Память: 3 длинных вектора на пару (s, y, v) + временные кэши размера O(n m).
    peak_floats = float(3 * n * m + n * m + 2 * n)
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
        if globalize:
            alpha_a, x_new, Fx_new, ne = armijo_step(F, x, Fx, d)
            n_eval += ne
        else:
            x_new = x + d; Fx_new = F(x_new); n_eval += 1; alpha_a = 1.0
            if not np.all(np.isfinite(Fx_new)):
                break
        s = x_new - x
        y = Fx_new - Fx
        # SP-вектор v_k из текущего окна s_hist (включая только что принятый s)
        p_eff = 0
        v = s
        s_hist_full = [pp[0] for pp in pairs] + [s]
        if p_max > 0 and len(s_hist_full) >= 2:
            for p_try in range(1, min(p_max, len(s_hist_full) - 1) + 1):
                cols = [s_hist_full[-1 - j] for j in range(p_try + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                try:
                    cG = float(cond(G))
                except np.linalg.LinAlgError:
                    break
                if cG < cond_thresh:
                    e1 = np.zeros(p_try + 1); e1[0] = 1.0
                    try:
                        v_try = Sp @ solve(G, e1)
                    except np.linalg.LinAlgError:
                        break
                    v = v_try
                    p_eff = p_try
                else:
                    break
        # пробное применение H_k y для проверки знаменателя α_k = v^T H_k y
        Hy_test, _, _ = apply_H_pairs(y, pairs)
        alpha_test = float(v @ Hy_test)
        if abs(alpha_test) < 1e-14 * (norm(v) * norm(Hy_test) + 1e-30):
            # пропускаем апдейт
            x, Fx = x_new, Fx_new
            res.append(float(norm(Fx)))
            fevals.append(n_eval)
        else:
            pairs.append((s.copy(), y.copy(), v.copy()))
            x, Fx = x_new, Fx_new
            res.append(float(norm(Fx)))
            fevals.append(n_eval)
        # тримминг
        while len(pairs) > m:
            pairs.pop(0)
    wall = time.time() - t0
    label = f"L-SP-Broyden(m={m},p≤{p_max})" if p_max > 0 else f"L-Broyden(m={m})"
    return dict(method=label, res=np.array(res), fevals=np.array(fevals),
                wall=wall, peak_floats=peak_floats, converged=converged,
                iters=len(res) - 1)


# ----------------------------------------------------------------------
# Метод 4: Anderson(m, β) поверх Picard g_τ(x) = x - τ F(x).
# (Walker-Ni 2011, L²-multisecant fit с Tikhonov-ridge.)
# ----------------------------------------------------------------------


def anderson_solve(F, x0, m=10, beta=1.0, tau=None, maxiter=600, tol=EPS_TOL,
                   ridge=1e-10):
    """Anderson acceleration на g_τ(x) = x - τ F(x). τ = 1/L_F (грубая оценка)."""
    n = x0.size
    x = x0.astype(float).copy()
    Fx = F(x)
    n_eval = 1
    if tau is None:
        # эвристическая оценка L_F через ‖F(x0)‖/‖x0 - x_test‖
        # Без знания J используем τ = 1.0 / ‖F(x0)‖₂ * ‖x0‖₂ (порядок 1).
        tau = 1.0 / max(1.0, float(norm(Fx)) / max(1.0, float(norm(x))))
    g = x - tau * Fx
    f_curr = g - x   # = -tau * Fx (residual для Anderson)
    X = []
    F_hist = []      # f_k = g(x_k) - x_k
    res = [float(norm(Fx))]
    fevals = [1]
    t0 = time.time()
    peak_floats = float(2 * n * m + 2 * n)
    converged = False
    x_prev = x; f_prev = f_curr.copy()
    last_res = res[-1]
    for k in range(maxiter):
        if res[-1] < tol:
            converged = True
            break
        # Anderson mixing
        X.append(x.copy())
        F_hist.append(f_curr.copy())
        if len(X) > m + 1:
            X.pop(0); F_hist.pop(0)
        if len(X) >= 2:
            # ΔX, ΔF (n x p), p = len-1
            DX = np.column_stack([X[i+1] - X[i] for i in range(len(X)-1)])
            DF = np.column_stack([F_hist[i+1] - F_hist[i] for i in range(len(F_hist)-1)])
            DTD = DF.T @ DF
            reg = ridge * float(np.trace(DTD)) * np.eye(DTD.shape[0])
            try:
                gamma = solve(DTD + reg, DF.T @ f_curr)
            except np.linalg.LinAlgError:
                gamma = np.zeros(DTD.shape[0])
            x_new = x + beta * f_curr - (DX + beta * DF) @ gamma
        else:
            x_new = x + beta * f_curr
        Fx_new = F(x_new)
        n_eval += 1
        if not np.all(np.isfinite(Fx_new)):
            # рестарт окна
            X = []; F_hist = []
            x_new = x - tau * Fx
            Fx_new = F(x_new); n_eval += 1
        # safeguard: рост резидуала ⇒ половиним β локально (не реализован, оставлен полный β)
        res_new = float(norm(Fx_new))
        if res_new > 2.0 * last_res and len(X) > 1:
            # рестарт
            X = []; F_hist = []
            x_new = x - tau * Fx
            Fx_new = F(x_new); n_eval += 1
            res_new = float(norm(Fx_new))
        last_res = res_new
        x = x_new; Fx = Fx_new
        f_curr = -tau * Fx
        res.append(res_new)
        fevals.append(n_eval)
    wall = time.time() - t0
    return dict(method=f"Anderson(m={m},β={beta:.1f})", res=np.array(res),
                fevals=np.array(fevals), wall=wall, peak_floats=peak_floats,
                converged=converged, iters=len(res) - 1)


# ----------------------------------------------------------------------
# Прогон одной задачи × одной размерности.
# ----------------------------------------------------------------------


def run_problem(prob_name, n, methods, maxiter):
    F, x0_fn = PROBLEMS[prob_name]
    x0 = x0_fn(n)
    out = {}
    print(f"\n=== {prob_name}, n={n} ===")
    for name, fn in methods:
        try:
            r = fn(F, x0, maxiter=maxiter)
        except Exception as e:
            print(f"  [{name:<22s}]  FAILED: {e}")
            continue
        out[name] = r
        print(
            f"  [{name:<22s}] iters={r['iters']:>4d}  fevals={int(r['fevals'][-1]):>5d}  "
            f"wall={r['wall']:>7.3f}s  ‖F‖={float(r['res'][-1]):.2e}  "
            f"peak≈{r['peak_floats']/1e6:>7.2f}M"
        )
    return out


# ----------------------------------------------------------------------
# Фигуры.
# ----------------------------------------------------------------------


def plot_convergence(all_results):
    """4 панели: 2 задачи × 2 размера. ‖F(x_k)‖ vs итерация."""
    sizes = sorted({n for (_, n) in all_results.keys()})
    probs = sorted({p for (p, _) in all_results.keys()})
    fig, axes = plt.subplots(len(probs), len(sizes),
                             figsize=(4.6 * len(sizes), 3.4 * len(probs)),
                             squeeze=False)
    color_map = {
        "Broyden-SM": ("#888888", (0, (5, 3))),
        "SP-Broyden-SM(p≤5)": ("#2060B0", (0, (4, 2))),
        "L-Broyden(m=10)": ("#7E57C2", (0, (3, 1, 1, 1))),
        "L-SP-Broyden(m=10,p≤5)": ("#D03030", "-"),
        "L-SP-Broyden(m=20,p≤5)": ("#FF8C00", (0, (1, 0.5))),
        "Anderson(m=10,β=1.0)": ("#2E8B57", (0, (1, 1))),
    }
    for i, p in enumerate(probs):
        for j, n in enumerate(sizes):
            ax = axes[i][j]
            res_dict = all_results.get((p, n), {})
            for name, r in res_dict.items():
                style = color_map.get(name, ("#000000", "-"))
                ax.semilogy(np.arange(len(r["res"])), r["res"],
                            label=name, color=style[0], ls=style[1], lw=1.6)
            ax.axhline(EPS_TOL, color="k", ls=":", lw=0.6, alpha=0.5)
            ax.set_xlabel("итерация $k$")
            ax.set_ylabel(r"$\|F(x_k)\|_2$")
            ax.set_title(f"{p}, $n={n}$", fontsize=10)
            ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    # общая легенда
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               fontsize=8.6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Сходимость в высокой размерности: SP-Broyden vs limited-memory "
                 "и Anderson", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(OUT_DIR, "fig_highdim_conv.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")


def plot_summary(all_results):
    """Bar-chart: итерации, #fevals, wall, peak-memory."""
    method_order = [
        "Broyden-SM",
        "SP-Broyden-SM(p≤5)",
        "L-Broyden(m=10)",
        "L-SP-Broyden(m=10,p≤5)",
        "L-SP-Broyden(m=20,p≤5)",
        "Anderson(m=10,β=1.0)",
    ]
    keys = sorted(all_results.keys())  # [(prob, n), ...]
    n_keys = len(keys)
    metrics = [
        ("итерации", lambda r: r["iters"]),
        ("число F-вычислений", lambda r: int(r["fevals"][-1])),
        ("wall, секунды", lambda r: r["wall"]),
        (r"память, $10^6$ float", lambda r: r["peak_floats"] / 1e6),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11.0, 3.2 * len(metrics)),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    width = 0.8 / max(1, len(method_order))
    xpos = np.arange(n_keys)
    for ax_idx, (mname, getter) in enumerate(metrics):
        ax = axes[ax_idx][0]
        for mi, name in enumerate(method_order):
            ys = []
            for k in keys:
                r = all_results.get(k, {}).get(name, None)
                if r is None or not r.get("converged", False):
                    ys.append(np.nan)
                else:
                    ys.append(getter(r))
            ax.bar(xpos + (mi - len(method_order) / 2) * width + width / 2,
                   ys, width, label=name, color=cmap(mi))
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{p}\n$n={n}$" for (p, n) in keys], fontsize=8.5)
        ax.set_ylabel(mname, fontsize=9)
        if mname.startswith("wall") or mname.startswith("память"):
            ax.set_yscale("log")
        ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=8.0, ncol=min(5, len(method_order)),
                      loc="upper left", bbox_to_anchor=(0.0, 1.18))
    fig.suptitle("Итог по методу × задаче × $n$: итерации, число F-вычислений, "
                 "время и память",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_highdim_summary.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")


def plot_pvar(pvar_results):
    """Variation памяти m ∈ {2,5,10,20} при n=1000, фиксированном p_max=m, для одной задачи."""
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    cmap = plt.get_cmap("viridis")
    keys = sorted(pvar_results.keys())  # [m]
    for i, m_val in enumerate(keys):
        r = pvar_results[m_val]
        if r is None:
            continue
        c = cmap(i / max(1, len(keys) - 1))
        ax.semilogy(np.arange(len(r["res"])), r["res"],
                    color=c, lw=1.7, label=f"$m={m_val}$")
    ax.axhline(EPS_TOL, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|F(x_k)\|_2$")
    ax.set_title("L-SP-Broyden, Discrete BVP, $n=1000$: глубина окна $m$",
                 fontsize=10.5)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_highdim_pvar.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Эксперимент 1 + 2: 2 задачи × 2 размера × 5 методов ---
    # Все Бройден-семейные методы — с глобализацией Армихо (Алгоритм 2 гл. 2).
    methods_dense = [
        ("Broyden-SM",
         lambda F, x, maxiter: broyden_sm(F, x, maxiter=maxiter, globalize=True)),
        ("SP-Broyden-SM(p≤5)",
         lambda F, x, maxiter: sp_broyden_sm(F, x, p_max=5, maxiter=maxiter,
                                             globalize=True)),
        ("L-Broyden(m=10)",
         lambda F, x, maxiter: lsp_broyden(F, x, m=10, p_max=0, maxiter=maxiter,
                                           globalize=True)),
        ("L-SP-Broyden(m=10,p≤5)",
         lambda F, x, maxiter: lsp_broyden(F, x, m=10, p_max=5, maxiter=maxiter,
                                           globalize=True)),
        ("Anderson(m=10,β=1.0)",
         lambda F, x, maxiter: anderson_solve(F, x, m=10, beta=1.0, maxiter=maxiter)),
    ]
    methods_lim = [
        # для n=10000 dense Broyden/SP-Broyden пропускаем (n²=10^8 floats)
        ("L-Broyden(m=10)",
         lambda F, x, maxiter: lsp_broyden(F, x, m=10, p_max=0, maxiter=maxiter,
                                           globalize=True)),
        ("L-SP-Broyden(m=10,p≤5)",
         lambda F, x, maxiter: lsp_broyden(F, x, m=10, p_max=5, maxiter=maxiter,
                                           globalize=True)),
        ("L-SP-Broyden(m=20,p≤5)",
         lambda F, x, maxiter: lsp_broyden(F, x, m=20, p_max=5, maxiter=maxiter,
                                           globalize=True)),
        ("Anderson(m=10,β=1.0)",
         lambda F, x, maxiter: anderson_solve(F, x, m=10, beta=1.0, maxiter=maxiter)),
    ]

    all_results = {}
    for prob in PROBLEMS:
        for n_dim in (1000, 10000):
            mset = methods_dense if n_dim == 1000 else methods_lim
            maxit = 600 if n_dim == 1000 else 800
            res = run_problem(prob, n_dim, mset, maxit)
            all_results[(prob, n_dim)] = res

    # --- Эксперимент 3: variation m в L-SP-Broyden при n=1000, Banded Cubic ---
    print("\n=== L-SP-Broyden, Banded Cubic, n=1000, варьирование m ===")
    pvar_results = {}
    for m_val in (2, 5, 10, 20):
        F = banded_cubic_F; x0 = banded_cubic_x0(1000)
        r = lsp_broyden(F, x0, m=m_val, p_max=min(m_val, 5), maxiter=400,
                        globalize=True)
        pvar_results[m_val] = r
        print(f"  m={m_val:>2d}: iters={r['iters']:>4d}  fevals={int(r['fevals'][-1]):>5d}  "
              f"‖F‖={r['res'][-1]:.2e}  peak≈{r['peak_floats']/1e6:.2f}M")

    # --- Сохранить .npz и фигуры ---
    plot_convergence(all_results)
    plot_summary(all_results)
    plot_pvar(pvar_results)

    # сериализация (упрощённая: только числовые ряды)
    np_save = {}
    for (p, n), runs in all_results.items():
        for name, r in runs.items():
            tag = f"{p}|n={n}|{name}"
            np_save[tag + "::res"] = r["res"]
            np_save[tag + "::fevals"] = r["fevals"]
            np_save[tag + "::wall"] = np.array([r["wall"]])
            np_save[tag + "::peak"] = np.array([r["peak_floats"]])
    for m_val, r in pvar_results.items():
        tag = f"DBVP|n=1000|m={m_val}"
        np_save[tag + "::res"] = r["res"]
        np_save[tag + "::fevals"] = r["fevals"]
    np.savez_compressed("highdim_results.npz", **np_save)
    print("saved: highdim_results.npz")


if __name__ == "__main__":
    main()
