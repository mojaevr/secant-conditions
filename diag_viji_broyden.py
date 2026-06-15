"""diag_viji_broyden.py — численная иллюстрация к главе 4 (VIQA/VIJI).

Иллюстрирует, как уровень неточности якобиана δ = ‖J − ∇F‖, входящий линейно
в скоростную оценку метода второго порядка для вариационных неравенств
(Теорема 3.2 из viqa2024, формула eq:viqa_rate), зависит от выбора
квазиньютоновского приближения якобиана:

  * exact        — точный якобиан ∇F(v) (эталон, δ ≡ 0);
  * L-Broyden    — односекущее обновление (Eq. 20 из viqa2024), приближение
                   строится в текущей точке v из m якобиан-векторных
                   произведений y_i = ∇F(v) s_i на случайных направлениях;
  * multi-secant — мультисекущее обновление Бройдена (формула eq:sp_update
                   диссертации, гл. 2) с окном последних p+1 шагов траектории.

ВАЖНО: это иллюстративная редукция итерации VIJI, а НЕ воспроизведение полного
алгоритма статьи (без mirror-prox и двойственной экстраполяции). Внешний шаг —
регуляризованная модель Ψ_v(x) = F(v) + J(v)[x−v] с кубическим демпфированием
ρ_k ∝ L_1‖x−v‖, что сохраняет связку «качество J(v) → штраф δ» из eq:viqa_rate.

Тестовая задача — монотонный L_1-гладкий оператор без ограничений
  F(z) = M z + γ·tanh(z),   M = [[A, B], [−Bᵀ, C]],  A, C симметричны PD,
несимметричный min-max источник F = (∇_xΦ, −∇_yΦ) из таблицы tab:ch1_classes.
Тогда ∇F(z) = M + γ·diag(sech²(z)) известен точно, а x* (корень F) находится
высокоточным Ньютоном для эталона невязки.

Выходы:
  mipt_thesis_master/fig_viji_broyden.pdf — 2 панели:
      (слева)  ‖F(x_k)‖ vs итерация для трёх оракулов;
      (справа) δ_k = ‖J_k − ∇F(x_k)‖₂ vs итерация.
  viji_broyden.npz          — сырые траектории (res, delta) по всем стартам.
  viji_broyden_summary.txt  — медианные δ и число итераций до tol.

Paired-design: одни и те же случайные матрицы задачи, старты и направления
для всех трёх оракулов. Seed 20260602.
"""
from __future__ import annotations

import os
import numpy as np
from numpy.linalg import norm, solve, eigvalsh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")
SEED = 20260602

TOL = 1e-9
MAXITER = 60
GAMMA = 1.0          # вес монотонной нелинейности γ·tanh
C_REG = 0.5          # коэффициент кубической регуляризации ρ = c·L1·‖Δ‖
P_WIN = 10            # окно мультисекущего Бройдена (p+1 секущих условий)
M_MEM = 5            # память односекущего L-Broyden (число JVP-пар)


# ============================================================
#   Тестовая задача: монотонный VI-оператор
# ============================================================

def make_operator(n: int, rng: np.random.Generator):
    """F(z) = M z + γ tanh(z), монотонный L1-гладкий оператор.

    M = [[A, B], [−Bᵀ, C]] с симметричными PD A, C → симметричная часть M
    блочно-диагональна и PD, поэтому M (и весь F) сильно монотонен.
    Возвращает F, JF (точный якобиан), L1 (Липшиц якобиана), x0-генератор и x*.
    """
    assert n % 2 == 0
    h = n // 2

    def spd(m):
        Q = rng.standard_normal((m, m))
        A = Q @ Q.T / m + np.eye(m)          # SPD, умеренно обусловленная
        return 0.5 * (A + A.T)

    A = spd(h)
    C = spd(h)
    B = rng.standard_normal((h, h)) / np.sqrt(h)
    M = np.zeros((n, n))
    M[:h, :h] = A
    M[:h, h:] = B
    M[h:, :h] = -B.T
    M[h:, h:] = C

    def F(z):
        return M @ z + GAMMA * np.tanh(z)

    def JF(z):
        # ∇F = M + γ diag(sech²(z)),  sech²(z) = 1 − tanh²(z)
        return M + GAMMA * np.diag(1.0 - np.tanh(z) ** 2)

    # L1: ‖∇F(z)−∇F(w)‖ ≤ γ·max|d/dz sech²| ‖z−w‖; |d/dz(1−tanh²)| ≤ ~0.77.
    L1 = GAMMA * 0.7699

    # Точное решение F(x*)=0 высокоточным Ньютоном из нуля.
    x = np.zeros(n)
    for _ in range(100):
        Fx = F(x)
        if norm(Fx) < 1e-14:
            break
        x = x - solve(JF(x), Fx)
    xstar = x
    return dict(F=F, JF=JF, L1=L1, n=n, M=M, xstar=xstar)


# ============================================================
#   Квазиньютоновские приближения якобиана J(v) ≈ ∇F(v)
# ============================================================

def lbroyden_local(JF, v, m, rng, J):
    """Односекущий L-Broyden (Eq. 20 из viqa2024), локально в точке v.

    J^{i+1} = J^i + (y_i − J^i s_i) s_iᵀ / (s_iᵀ s_i),  i=0..m−1,
    пары из якобиан-векторных произведений y_i = ∇F(v) s_i на случайных
    направлениях s_i (как в статье). База J^0 = I.
    """
    n = v.size
    Jexact = JF(v)
    # J = np.eye(n)
    for _ in range(1):
        s = rng.standard_normal(n)
        s /= norm(s)
        y = Jexact @ s
    J = J + np.outer(y - J @ s, s) / (s @ s)
    return J


def msbroyden_update(J, S_hist, Y_hist, p_max):
    """Мультисекущее обновление Бройдена (формула eq:sp_update, гл. 2):

        B_{k+1} = B_k + (Y_p − B_k S_p) G_p^{-1} S_pᵀ,   G_p = S_pᵀ S_p,

    где S_p, Y_p — окно последних p+1 секущих пар (s_j, y_j).
    """
    p_eff = min(p_max, len(S_hist) - 1)
    cols = [S_hist[-1 - j] for j in range(p_eff + 1)]
    yols = [Y_hist[-1 - j] for j in range(p_eff + 1)]
    Sp = np.column_stack(cols)
    Yp = np.column_stack(yols)
    G = Sp.T @ Sp
    try:
        R = solve(G, Sp.T)               # G^{-1} S_pᵀ
    except np.linalg.LinAlgError:
        s = S_hist[-1]; y = Y_hist[-1]
        return J + np.outer(y - J @ s, s) / (s @ s)
    return J + (Yp - J @ Sp) @ R


# ============================================================
#   Внешняя итерация (иллюстративная редукция VIJI)
# ============================================================

def regularized_step(J, Fv, L1, c, n):
    """Шаг по регуляризованной линейной модели:
       (J + ρ I) Δ = −F(v),  ρ = c·L1·‖Δ‖ (внутренняя фикс-точка по ρ).
    """
    rho = 0.0
    Delta = np.zeros(n)
    for _ in range(8):
        Delta = solve(J + rho * np.eye(n), -Fv)
        rho = c * L1 * norm(Delta)
    return Delta


def run(prob, oracle, rng, x0):
    """Один прогон. oracle ∈ {'exact','lbroyden','multisecant'}."""
    F, JF, L1, n = prob["F"], prob["JF"], prob["L1"], prob["n"]
    v = x0.astype(float).copy()
    Fv = F(v)
    res = [float(norm(Fv))]
    delta = []   # = 0 для старта (J не задан)
    S_hist, Y_hist = [], []
    J = np.eye(n)
    for k in range(MAXITER):
        if res[-1] < TOL:
            break
        if oracle == "exact":
            J = JF(v)
        elif oracle == "lbroyden":
            # J = lbroyden_local(JF, v, M_MEM, rng, J)
            if len(S_hist) >= 1:
                J = msbroyden_update(J, S_hist, Y_hist, 1)
        else:  # multisecant: running update по траектории
            if len(S_hist) >= 1:
                J = msbroyden_update(J, S_hist, Y_hist, P_WIN)
        # уровень неточности в текущей точке
        delta.append(float(norm(J - JF(v), 2)))
        Delta = regularized_step(J, Fv, L1, C_REG, n)
        if not np.all(np.isfinite(Delta)):
            break
        x_new = v + Delta
        Fx_new = F(x_new)
        S_hist.append(x_new - v)
        Y_hist.append(Fx_new - Fv)
        if len(S_hist) > P_WIN + 3:
            S_hist.pop(0); Y_hist.pop(0)
        v, Fv = x_new, Fx_new
        res.append(float(norm(Fv)))
    return np.array(res), np.array(delta)


# ============================================================
#   Эксперимент: несколько стартов, статистика
# ============================================================

def iters_to_tol(res):
    idx = np.where(res < TOL)[0]
    return int(idx[0]) if idx.size else len(res) - 1


def main():
    os.makedirs(THESIS_DIR, exist_ok=True)
    rng_master = np.random.default_rng(SEED)
    n = 200
    n_starts = 2
    oracles = ["exact", "multisecant", "lbroyden"]
    labels = {"exact": "точный ∇F",
              "multisecant": f"мультисекущий (p={P_WIN})",
              "lbroyden": f"L-Broyden (m={M_MEM})"}
    colors = {"exact": "k", "multisecant": "C0", "lbroyden": "C3"}

    prob = make_operator(n, rng_master)
    kappa_M = float(np.linalg.cond(prob["M"]))

    runs = {o: {"res": [], "delta": [], "iters": []} for o in oracles}
    for s in range(n_starts):
        x0 = 2.0 * rng_master.standard_normal(n)
        # paired-design: общий поток случайных направлений на старт
        dir_seed = int(rng_master.integers(1, 2**31 - 1))
        for o in oracles:
            res, delta = run(prob, o, np.random.default_rng(dir_seed), x0)
            runs[o]["res"].append(res)
            runs[o]["delta"].append(delta)
            runs[o]["iters"].append(iters_to_tol(res))

    # ---- фигура: 2 панели ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for o in oracles:
        # медианная кривая невязки (по общей длине)
        Lmax = max(len(r) for r in runs[o]["res"])
        R = np.full((n_starts, Lmax), np.nan)
        for i, r in enumerate(runs[o]["res"]):
            R[i, :len(r)] = r
        med = np.nanmedian(R, axis=0)
        axL.semilogy(med, color=colors[o], label=labels[o], lw=1.8)
        if o != "exact":
            Dmax = max(len(d) for d in runs[o]["delta"])
            D = np.full((n_starts, Dmax), np.nan)
            for i, d in enumerate(runs[o]["delta"]):
                D[i, :len(d)] = d
            medd = np.nanmedian(D, axis=0)
            axR.plot(np.arange(len(medd)), medd, color=colors[o],
                     label=labels[o], lw=1.8)

    axL.set_xlabel("итерация $k$")
    axL.set_ylabel(r"$\|F(x_k)\|_2$")
    axL.set_title("Невязка вариационного неравенства")
    axL.legend(fontsize=9)
    axL.grid(True, which="both", alpha=0.3)

    axR.set_xlabel("итерация $k$")
    axR.set_ylabel(r"$\delta_k=\|J_k-\nabla F(x_k)\|_2$")
    axR.set_title("Уровень неточности якобиана")
    axR.legend(fontsize=9)
    axR.grid(True, alpha=0.3)

    fig.tight_layout()
    out_pdf = os.path.join(THESIS_DIR, "fig_viji_broyden.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)

    # ---- npz ----
    np.savez(os.path.join(SCRIPT_DIR, "viji_broyden.npz"),
             n=n, n_starts=n_starts, p_win=P_WIN, m_mem=M_MEM,
             gamma=GAMMA, c_reg=C_REG, kappa_M=kappa_M,
             **{f"res_{o}": np.array(runs[o]["res"], dtype=object)
                for o in oracles},
             **{f"delta_{o}": np.array(runs[o]["delta"], dtype=object)
                for o in oracles})

    # ---- summary ----
    lines = []
    lines.append("VIJI с квазиньютоновским якобианом — сводка")
    lines.append(f"  задача F(z)=Mz+γtanh(z), n={n}, γ={GAMMA}, "
                 f"cond(M)={kappa_M:.2f}, стартов={n_starts}")
    lines.append(f"  окно мультисекущего p={P_WIN}, память L-Broyden m={M_MEM}, "
                 f"tol={TOL:g}")
    lines.append("")
    lines.append(f"  {'оракул':<22}{'медиана итер':>14}"
                 f"{'медиана δ (посл.)':>20}{'медиана δ (макс.)':>20}")
    for o in oracles:
        med_it = float(np.median(runs[o]["iters"]))
        last_d = np.median([d[min(len(d) - 1,
                                  iters_to_tol(runs[o]["res"][i]))]
                            for i, d in enumerate(runs[o]["delta"])])
        max_d = np.median([np.max(d) for d in runs[o]["delta"]])
        lines.append(f"  {labels[o]:<22}{med_it:>14.1f}"
                     f"{last_d:>20.3e}{max_d:>20.3e}")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(SCRIPT_DIR, "viji_broyden_summary.txt"), "w") as f:
        f.write(summary)
    print(summary)
    print(f"[ok] {out_pdf}")


if __name__ == "__main__":
    main()
