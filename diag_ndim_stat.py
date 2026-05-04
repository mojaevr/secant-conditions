"""diag_ndim_stat.py — статистическая версия экспериментов главы 3 в n>=10.

SS-конструкция (Definition def:ss) реализуется в виде индефинитной коррекции
σ u u^T и предназначена для базовых обновлений, которые сами по себе НЕ
сохраняют PD: в данной диссертации это SR1 и PSB. Сравнения BFGS/DFP с их
«SS-вариантами» исключены: BFGS (PD-сохраняющий) теряет PD после σuu^T-коррекции,
а классический DFP формулируется в H-форме и не имеет естественной B-формы для
проекции R'_k = B' S_p - Y_p.

Выходные фигуры:
  fig_ndim_stat_sr1.pdf        — SS-SR1 vs SR1, сходимость ||grad f||
  fig_ndim_stat_psb.pdf        — SS-PSB vs PSB, сходимость ||grad f||
  fig_ndim_stat_rpast_psb.pdf  — SS-PSB vs PSB, ||B_k S_p - Y_p||_F
                                  (прямая иллюстрация теоремы ss_props п. 3)
Метрика ||B_k S_p - Y_p||_F сохраняется в npz для всех пар, но фигура строится
только для PSB-пары: на SR1-паре rpast-разрыв нечитаем визуально, поскольку
сам SR1 на использованных режимах часто не деградирует достаточно сильно
(см. par:ssn-rpast в ch3_symmetric.tex).

Для каждой пары — 5 панелей в сетке 2x3 (последняя ячейка скрыта):
  верхний ряд: Rosenbrock chained (n=10), Extended Rosenbrock (n=10) — гладкий контроль
  нижний ряд: ECV(α=100, β=1.5) при n ∈ {10, 20, 50}                  — стиффовый режим

Все методы используют общий backtracking-Armijo по f (c1=1e-4, alpha<-alpha/2),
B_0 = I, окно прошлых секущих p = 5, критерий остановки ||g||_2 <= 1e-8 или
max_iter = 500. Поиск направления — через solve(B, -g).

Выходы:
  mipt_thesis_master/fig_ndim_stat_{sr1,psb}.pdf
  ndim_stat.npz                                 — сырые траектории и success-rate
  ndim_stat_summary.txt                         — таблица success rate / median iters

Раздельный seed для направлений: 20260503 (paired-design: одни и те же 50
случайных направлений на S^{n-1} используются всеми методами).
"""
from __future__ import annotations

import os
import pickle
import time
import numpy as np
from numpy.linalg import norm, solve, eigh, LinAlgError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")
CHECKPOINT = os.path.join(SCRIPT_DIR, ".ndim_stat_checkpoint.pkl")


# ============================================================
#   Тестовые задачи
# ============================================================

def extended_rosenbrock(n=10):
    assert n % 2 == 0
    def f(x):
        s = 0.0
        for i in range(0, n, 2):
            s += 100.0*(x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
        return s
    def g(x):
        gx = np.zeros(n)
        for i in range(0, n, 2):
            gx[i]   = -400.0*x[i]*(x[i+1] - x[i]**2) - 2.0*(1.0 - x[i])
            gx[i+1] = 200.0*(x[i+1] - x[i]**2)
        return gx
    x0 = np.empty(n); x0[0::2] = -1.2; x0[1::2] = 1.0
    return dict(name="Extended Rosenbrock", short="ext_rosen",
                n=n, f=f, g=g, x0=x0, fstar=0.0)


def rosenbrock_chained(n=10):
    def f(x):
        s = 0.0
        for i in range(n-1):
            s += 100.0*(x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
        return s
    def g(x):
        gx = np.zeros(n)
        for i in range(n-1):
            t = x[i+1] - x[i]**2
            gx[i]   += -400.0*x[i]*t - 2.0*(1.0 - x[i])
            gx[i+1] += 200.0*t
        return gx
    x0 = np.empty(n); x0[0::2] = -1.2; x0[1::2] = 1.0
    return dict(name="Rosenbrock chained", short="ros_chain",
                n=n, f=f, g=g, x0=x0, fstar=0.0)


def extended_curved_valley(n=10, alpha=100.0, beta=1.5):
    assert n % 2 == 0
    def f(x):
        s = 0.0
        for j in range(0, n, 2):
            t = x[j+1] - beta*x[j]**2
            s += 0.5*alpha*t*t + 0.5*x[j]**2 + 0.5*x[j+1]**2
        return s
    def g(x):
        gx = np.zeros(n)
        for j in range(0, n, 2):
            t = x[j+1] - beta*x[j]**2
            gx[j]   = x[j]   - 2.0*alpha*beta*x[j]*t
            gx[j+1] = x[j+1] + alpha*t
        return gx
    x0 = np.empty(n); x0[0::2] = 1.8; x0[1::2] = 1.8
    return dict(name=f"Extended Curved Valley (a={alpha:g}, b={beta:g})",
                short=f"ecv_n{n}", n=n, f=f, g=g, x0=x0, fstar=0.0)


# ============================================================
#   Линейный поиск: backtracking-Armijo по f
# ============================================================

def armijo_f(f, x, fx, g, d, c1=1e-4, alpha0=1.0, max_bt=40):
    gd = float(g @ d)
    if gd >= 0:
        return None, 0
    a = alpha0
    nfev = 0
    for _ in range(max_bt):
        nfev += 1
        if f(x + a*d) <= fx + c1*a*gd:
            return a, nfev
        a *= 0.5
    return a, nfev


# ============================================================
#   Базовые QN-обновления (B-форма; все удовлетворяют B_{k+1} s_k = y_k)
# ============================================================

def sr1_step(B, s, y, eps_skip=1e-8):
    r = y - B @ s
    den = float(r @ s)
    if abs(den) < eps_skip * max(norm(r)*norm(s), 1e-30):
        return B
    return B + np.outer(r, r) / den


def psb_step(B, s, y):
    r = y - B @ s
    ss = float(s @ s)
    if ss < 1e-30:
        return B
    rs = float(r @ s)
    return B + (np.outer(r, s) + np.outer(s, r))/ss - rs*np.outer(s, s)/(ss*ss)


# ============================================================
#   SS-коррекция (общая для всех базовых методов)
# ============================================================

def _ss_correct(Bp, s, S_window, Y_window):
    """MS-коррекция: B_{k+1} = B' + σ u u^T,  u ⊥ s_k.

    Шаги (Алгоритм \\ref{alg:ss_sr1}, шаги 2--4):
      R'_k = B' S_p - Y_p
      M'_k = (R'_k S_p^T + S_p R'_k^T) / 2
      A_k  = P M'_k P,  P = I - ŝ_k ŝ_k^T
      u    = ведущий собств. вектор A_k по |λ|, ⊥ s_k, ||u||=1
      σ    = -(u^T R'_k S_p^T u) / ||S_p^T u||^2
    Возвращает B'_k без коррекции при срабатывании любого из safeguard'ов
    (вырождение u, малость ||S_p^T u||^2, ошибка eigh)."""
    n = Bp.shape[0]
    if S_window is None or S_window.shape[1] == 0:
        return Bp
    R = Bp @ S_window - Y_window
    M = 0.5*(R @ S_window.T + S_window @ R.T)
    s_hat = s / max(norm(s), 1e-30)
    P = np.eye(n) - np.outer(s_hat, s_hat)
    A = P @ M @ P
    A = 0.5*(A + A.T)
    try:
        w, V = eigh(A)
    except LinAlgError:
        return Bp
    j = int(np.argmax(np.abs(w)))
    u = V[:, j]
    u = u - (u @ s_hat)*s_hat
    nu = norm(u)
    if nu < 1e-12:
        return Bp
    u /= nu
    Stu = S_window.T @ u
    den = float(Stu @ Stu)
    if den < 1e-15:
        return Bp
    sigma = -float(u @ (R @ Stu)) / den
    return Bp + sigma*np.outer(u, u)


def ss_sr1_step(B, s, y, S_window, Y_window, eps_skip=1e-8):
    return _ss_correct(sr1_step(B, s, y, eps_skip), s, S_window, Y_window)


def ss_psb_step(B, s, y, S_window, Y_window):
    return _ss_correct(psb_step(B, s, y), s, S_window, Y_window)


# ============================================================
#   Solver
# ============================================================

# Все методы — B-формы, направление d_k = -B_k^{-1} g_k.
BASE_METHODS  = {'sr1', 'psb'}
SS_METHODS    = {'ss_sr1', 'ss_psb'}
ALL_METHODS   = BASE_METHODS | SS_METHODS


def run(prob, x0, method, p_window=5, max_iter=500, tol=1e-8):
    if method not in ALL_METHODS:
        raise ValueError(method)
    n = prob['n']
    f, gfun = prob['f'], prob['g']
    x = x0.copy()
    B = np.eye(n)

    g = gfun(x); fx = f(x)
    n_f, n_g = 1, 1
    hist_g = [norm(g)]
    hist_calls = [n_g]
    # ||B S_p - Y_p||_F определена только когда в окне ≥1 ПРОШЛАЯ секущая,
    # то есть m_buf ≥ 2. На k=0 (до шагов) и k=1 (текущая секущая
    # удовлетворена точно по построению) метрика бессодержательна — NaN.
    hist_rpast = [float('nan')]
    S_buf = np.zeros((n, p_window))
    Y_buf = np.zeros((n, p_window))
    m_buf = 0
    converged = (hist_g[-1] <= tol)
    status = "converged" if converged else "max_iter"

    for k in range(max_iter):
        if hist_g[-1] <= tol:
            converged = True; status = "converged"; break

        try:
            d = solve(B, -g)
        except LinAlgError:
            d = -g
        if not np.all(np.isfinite(d)) or g @ d >= 0:
            d = -g

        a, nf_ls = armijo_f(f, x, fx, g, d)
        n_f += nf_ls
        if a is None:
            d = -g
            a, nf_ls = armijo_f(f, x, fx, g, d)
            n_f += nf_ls
            if a is None:
                status = "ls_fail"; break

        s = a*d
        x_new = x + s
        if not np.all(np.isfinite(x_new)) or norm(x_new) > 1e10:
            status = "diverge"; break
        g_new = gfun(x_new); n_g += 1
        f_new = f(x_new);    n_f += 1
        y = g_new - g

        if m_buf < p_window:
            S_buf[:, m_buf] = s; Y_buf[:, m_buf] = y; m_buf += 1
        else:
            S_buf[:, :-1] = S_buf[:, 1:]; S_buf[:, -1] = s
            Y_buf[:, :-1] = Y_buf[:, 1:]; Y_buf[:, -1] = y

        Sw = S_buf[:, :m_buf]; Yw = Y_buf[:, :m_buf]
        if   method == 'sr1':     B = sr1_step(B, s, y)
        elif method == 'psb':     B = psb_step(B, s, y)
        elif method == 'ss_sr1':  B = ss_sr1_step(B, s, y, Sw, Yw)
        elif method == 'ss_psb':  B = ss_psb_step(B, s, y, Sw, Yw)

        # Невязка прошлых секущих после обновления B (теорема ss_props п. 3).
        # Текущая колонка (s_k, y_k) всегда удовлетворена точно (B_{k+1} s_k = y_k);
        # ненулевой вклад дают только прошлые секущие. Если в окне ровно одна
        # текущая секущая (m_buf=1), метрика тривиально нулевая и неинформативна
        # — отмечаем NaN, чтобы исключить из графика.
        if m_buf > 1 and np.all(np.isfinite(B)):
            try:
                rpast = float(np.linalg.norm(B @ Sw - Yw, 'fro'))
            except (FloatingPointError, ValueError):
                rpast = float('nan')
        else:
            rpast = float('nan')
        hist_rpast.append(rpast)

        x, g, fx = x_new, g_new, f_new
        hist_g.append(norm(g))
        hist_calls.append(n_g)

    return dict(method=method, problem=prob['short'], n=n,
                converged=converged, status=status,
                iters=len(hist_g)-1, n_f=n_f, n_g=n_g,
                hist_g=np.asarray(hist_g),
                hist_calls=np.asarray(hist_calls),
                hist_rpast=np.asarray(hist_rpast),
                g_final=hist_g[-1])


# ============================================================
#   Эксперимент: 50 случайных направлений на сфере R_0 = ||x_0_std||
# ============================================================

def run_problem(prob, methods, n_dirs, U, max_iter, tol,
                raw=None, key_prefix=None, deadline=None):
    """U: (n_dirs, n) массив единичных направлений (paired-design).

    Если переданы (raw, key_prefix, deadline), результат пишется в raw[(key_prefix, mk)]
    после каждого метода и при превышении deadline возвращается частичный out."""
    R0 = norm(prob['x0'])
    out = {}
    for mk in methods:
        if raw is not None and key_prefix is not None and (*key_prefix, mk) in raw:
            out[mk] = raw[(*key_prefix, mk)]
            print(f"  [{mk:8s}] кэш ({len(out[mk])} стартов)")
            continue
        trajs = []
        t0 = time.time()
        n_conv = 0
        for di in range(n_dirs):
            x0 = R0 * U[di]
            tr = run(prob, x0, mk, max_iter=max_iter, tol=tol)
            trajs.append(tr)
            if tr['converged']:
                n_conv += 1
        dt = time.time() - t0
        out[mk] = trajs
        med_iters = int(np.median([t['iters'] for t in trajs if t['converged']])
                        if n_conv > 0 else -1)
        med_calls = int(np.median([t['n_g'] for t in trajs if t['converged']])
                        if n_conv > 0 else -1)
        print(f"  [{mk:8s}] {n_conv:2d}/{n_dirs} сошлось"
              f"  med_iter={med_iters:4d}  med_∇f={med_calls:4d}"
              f"  ({dt:5.1f}s)")
        if raw is not None and key_prefix is not None:
            raw[(*key_prefix, mk)] = trajs
            with open(CHECKPOINT, "wb") as fp:
                pickle.dump((raw, _R0_TABLE_REF[0]), fp)
        if deadline is not None and time.time() > deadline:
            return out, R0, False
    return out, R0, True


# Хак для прокидывания R0_table в чекпоинт (run_problem не имеет к нему доступ).
_R0_TABLE_REF = [None]


def aggregate_g(trajs, max_k):
    """Pad ||grad|| trajectories to common length max_k+1; pad-by-last."""
    n = len(trajs)
    G = np.full((n, max_k+1), np.nan)
    for i, tr in enumerate(trajs):
        L = len(tr['hist_g'])
        L = min(L, max_k+1)
        G[i, :L] = tr['hist_g'][:L]
        if L < max_k+1:
            G[i, L:] = tr['hist_g'][L-1]
    return G


def aggregate_rpast(trajs, max_k):
    """Pad ||B_k S_p - Y_p||_F trajectories; pad-by-last (как aggregate_g).

    Начало траектории (m_buf<2) хранится как NaN — оставляем NaN, на лог-графике
    эти точки просто не рисуются. В хвосте (k>L) копируем последнее НЕ-NaN
    значение, чтобы расходящиеся/рано-сошедшиеся траектории не съели медиану."""
    n = len(trajs)
    R = np.full((n, max_k+1), np.nan)
    for i, tr in enumerate(trajs):
        if 'hist_rpast' not in tr:
            continue  # старые trajectory dicts (без поля) — пропускаем
        h = np.asarray(tr['hist_rpast'])
        L = min(len(h), max_k+1)
        R[i, :L] = h[:L]
        if L < max_k+1:
            mask = np.isfinite(h[:L])
            if mask.any():
                last_val = h[:L][mask][-1]
                R[i, L:] = last_val
    return R


# ============================================================
#   Plot helpers
# ============================================================

# В каждой паре: SS-X — красная толстая (протагонист);
# X (оригинал) — синяя пунктирная.  Цвета совпадают для всех 4 фигур,
# поскольку графики автономны.
PAIR_STYLE = {
    'ss': dict(color='tab:red',  ls='-',  lw_med=2.4, z=6),
    'bs': dict(color='tab:blue', ls='--', lw_med=1.6, z=4),
}
LABEL = {
    'sr1': 'SR1',  'ss_sr1': 'SS-SR1',
    'psb': 'PSB',  'ss_psb': 'SS-PSB',
}


def plot_pair_panel(ax, results, base, ss, n_dirs, max_iter, tol, title):
    """Spaghetti + median по итерациям для одной (задача, n) и одной пары."""
    for mk, role in [(ss, 'ss'), (base, 'bs')]:
        trajs = results[mk]
        n_conv = sum(1 for t in trajs if t['converged'])
        st = PAIR_STYLE[role]
        G = aggregate_g(trajs, max_iter)
        ks = np.arange(G.shape[1])
        med = np.nanmedian(G, axis=0)
        for i in range(G.shape[0]):
            ax.plot(ks, G[i], color=st['color'], lw=0.5, alpha=0.16,
                    zorder=st['z'])
        ax.plot(ks, med, color=st['color'], lw=st['lw_med'], ls=st['ls'],
                label=f"{LABEL[mk]} ({n_conv}/{n_dirs})",
                zorder=st['z']+1)
    ax.set_yscale('log')
    ax.axhline(tol, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_xlabel(r"итерация $k$")
    ax.set_ylabel(r"$\|\nabla f(x_k)\|_2$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)


def plot_pair_panel_rpast(ax, results, base, ss, n_dirs, max_iter, title):
    """Median + IQR (q25-q75) для ||B_k S_p - Y_p||_F по обоим методам пары.

    Все 50 траекторий участвуют в медиане (включая расходящиеся —
    реализовано через pad-by-last в aggregate_rpast)."""
    floor = 1e-16  # клип снизу для логарифма (на сошедшихся к 0)
    for mk, role in [(ss, 'ss'), (base, 'bs')]:
        trajs = results[mk]
        n_conv = sum(1 for t in trajs if t['converged'])
        st = PAIR_STYLE[role]
        R = aggregate_rpast(trajs, max_iter)
        ks = np.arange(R.shape[1])
        # клипуем снизу, чтобы не уйти на -inf при идеально нулевых значениях
        Rc = np.clip(R, floor, None)
        med = np.nanmedian(Rc, axis=0)
        q25 = np.nanpercentile(Rc, 25, axis=0)
        q75 = np.nanpercentile(Rc, 75, axis=0)
        ax.fill_between(ks, q25, q75, color=st['color'], alpha=0.18,
                        zorder=st['z']-1, linewidth=0)
        ax.plot(ks, med, color=st['color'], lw=st['lw_med'], ls=st['ls'],
                label=f"{LABEL[mk]} ({n_conv}/{n_dirs})",
                zorder=st['z']+1)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.25)
    ax.set_xlabel(r"итерация $k$")
    ax.set_ylabel(r"$\|B_k S_p - Y_p\|_F$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9, loc='best', framealpha=0.85)


def make_pair_figure_rpast(raw, base, ss, ecv_configs, mgh_problems_n10,
                           n_dirs, max_iter, out_path):
    """Фигура для пары (SS-X, X), метрика — ||B_k S_p - Y_p||_F.
    Та же сетка 5 панелей (2x3 с пустой [0,2])."""
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.27)

    titles_mgh = {'ros_chain': r"Rosenbrock chained ($n=10$)",
                  'ext_rosen': r"Extended Rosenbrock ($n=10$)"}
    for j, short in enumerate(mgh_problems_n10):
        ax = fig.add_subplot(gs[0, j])
        results = {base: raw[(short, 10, base)],
                   ss:   raw[(short, 10, ss)]}
        plot_pair_panel_rpast(ax, results, base, ss, n_dirs, max_iter,
                              title=titles_mgh[short])

    for j, (nv, alpha, beta) in enumerate(ecv_configs):
        ax = fig.add_subplot(gs[1, j])
        results = {base: raw[('ecv', nv, base)],
                   ss:   raw[('ecv', nv, ss)]}
        plot_pair_panel_rpast(
            ax, results, base, ss, n_dirs, max_iter,
            title=fr"ECV ($n={nv}$, $\alpha={alpha:g}$, $\beta={beta:g}$)"
        )

    fig.suptitle(
        f"{LABEL[ss]} vs {LABEL[base]}: накопленная невязка прошлых секущих "
        r"$\|B_k S_p - Y_p\|_F$. "
        f"{n_dirs} стартов; медиана (жирная) и IQR (q25-q75, заливка)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")


def make_pair_figure(raw, base, ss, ecv_configs, mgh_problems_n10,
                     n_dirs, max_iter, tol, out_path):
    """Одна фигура на пару (SS-X, X): 5 панелей в сетке 2x3
    (упёрто, ячейка [0,2] скрыта)."""
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.27)

    # ---- Top row: MGH n=10 (2 панели; центр+левая) ----
    titles_mgh = {'ros_chain': r"Rosenbrock chained ($n=10$)",
                  'ext_rosen': r"Extended Rosenbrock ($n=10$)"}
    for j, short in enumerate(mgh_problems_n10):
        ax = fig.add_subplot(gs[0, j])
        results = {base: raw[(short, 10, base)],
                   ss:   raw[(short, 10, ss)]}
        plot_pair_panel(ax, results, base, ss, n_dirs, max_iter, tol,
                        title=titles_mgh[short])

    # ---- Bottom row: ECV n=10/20/50 ----
    for j, (nv, alpha, beta) in enumerate(ecv_configs):
        ax = fig.add_subplot(gs[1, j])
        results = {base: raw[('ecv', nv, base)],
                   ss:   raw[('ecv', nv, ss)]}
        plot_pair_panel(
            ax, results, base, ss, n_dirs, max_iter, tol,
            title=fr"ECV ($n={nv}$, $\alpha={alpha:g}$, $\beta={beta:g}$)"
        )

    fig.suptitle(
        f"{LABEL[ss]} vs {LABEL[base]}: {n_dirs} случайных стартов "
        r"$\|x_0\|=\mathrm{const}$ (тонкие линии) и медиана; "
        r"в скобках — успехи/всего",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")


# ============================================================
#   main
# ============================================================

def main():
    rng = np.random.default_rng(20260503)
    n_dirs = 50
    max_iter = 500
    tol = 1e-8

    # 4 метода гоняем на всех конфигурациях (paired-design):
    methods_all = ['sr1', 'ss_sr1', 'psb', 'ss_psb']
    pairs = [('sr1', 'ss_sr1'),
             ('psb', 'ss_psb')]

    ecv_configs = [(10, 100.0, 1.5),
                   (20, 100.0, 1.5),
                   (50, 100.0, 1.5)]
    mgh_problems_n10 = ['ros_chain', 'ext_rosen']

    # ---- Direction tables (одни и те же для всех методов) ----
    U_by_n = {}
    needed_ns = sorted({nv for nv, _, _ in ecv_configs} | {10})
    for nv in needed_ns:
        Z = rng.standard_normal((n_dirs, nv))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        U_by_n[nv] = Z

    # ---- Resume from checkpoint if exists ----
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "rb") as fp:
            raw, R0_table = pickle.load(fp)
        # Инвалидация: (i) траектории без поля hist_rpast; (ii) траектории
        # старого формата с placeholder hist_rpast[0]=0.0 (теперь NaN).
        bad_keys = []
        for k, trajs in raw.items():
            if not trajs:
                continue
            t0 = trajs[0]
            if 'hist_rpast' not in t0:
                bad_keys.append(k); continue
            hp0 = t0['hist_rpast'][0] if len(t0['hist_rpast']) > 0 else None
            if hp0 is not None and not np.isnan(hp0):
                bad_keys.append(k)
        for k in bad_keys:
            del raw[k]
        if bad_keys:
            print(f"checkpoint: invalidated {len(bad_keys)} entries старого формата "
                  f"(пересчитаются)")
        print(f"resumed from checkpoint: {len(raw)} (config, method) entries")
    else:
        raw = {}  # (problem_short, n, method) -> list[traj dict]
        R0_table = {}
    _R0_TABLE_REF[0] = R0_table

    deadline = time.time() + 38.0  # бюджет одного запуска (sandbox bash timeout 45s)

    # ---- ECV (3 dims) ----
    print("=" * 68)
    print("Эксперимент 1: Extended Curved Valley, alpha=100, beta=1.5")
    print("=" * 68)
    for nv, alpha, beta in ecv_configs:
        if all(('ecv', nv, mk) in raw for mk in methods_all):
            print(f"[ECV n={nv}] уже посчитано, пропускаю")
            continue
        if time.time() > deadline:
            break
        prob = extended_curved_valley(n=nv, alpha=alpha, beta=beta)
        print(f"\n[ECV n={nv}]  R0=||x_0||={norm(prob['x0']):.3f}")
        out, R0, _ = run_problem(prob, methods_all, n_dirs, U_by_n[nv],
                                 max_iter, tol, raw=raw,
                                 key_prefix=('ecv', nv), deadline=deadline)
        R0_table[('ecv', nv)] = R0

    # ---- MGH (2 problems, n=10) ----
    print()
    print("=" * 68)
    print("Эксперимент 2: MGH (Rosenbrock chained, Extended Rosenbrock), n=10")
    print("=" * 68)
    for short in mgh_problems_n10:
        if all((short, 10, mk) in raw for mk in methods_all):
            print(f"[{short}] уже посчитано, пропускаю")
            continue
        if time.time() > deadline:
            break
        prob = (rosenbrock_chained(10) if short == 'ros_chain'
                else extended_rosenbrock(10))
        print(f"\n[{prob['name']}]  R0=||x_0||={norm(prob['x0']):.3f}")
        out, R0, _ = run_problem(prob, methods_all, n_dirs, U_by_n[10],
                                 max_iter, tol, raw=raw,
                                 key_prefix=(short, 10), deadline=deadline)
        R0_table[(short, 10)] = R0

    # ---- Все ли конфиги собраны? Если нет — выходим, чтобы продолжить ----
    expected_configs = ([('ecv', nv) for nv, _, _ in ecv_configs]
                        + [(s, 10) for s in mgh_problems_n10])
    finished_all = all(all((short, nv, mk) in raw for mk in methods_all)
                       for short, nv in expected_configs)
    if not finished_all:
        print()
        print(f">>> НЕ ВСЕ КОНФИГИ СОБРАНЫ; чекпоинт сохранён в {CHECKPOINT}")
        print(f">>> запустите скрипт снова, чтобы продолжить")
        return

    # ============================================================
    #   4 фигуры: по 2 на пару (SS-X, X) — convergence + rpast
    # ============================================================
    fig_specs = [('sr1', 'ss_sr1', 'fig_ndim_stat_sr1.pdf'),
                 ('psb', 'ss_psb', 'fig_ndim_stat_psb.pdf')]
    print()
    for base, ss, fname in fig_specs:
        out_path = os.path.join(THESIS_DIR, fname)
        make_pair_figure(raw, base, ss, ecv_configs, mgh_problems_n10,
                         n_dirs, max_iter, tol, out_path)

    # rpast figure: только PSB-пара (SR1-пара в главу не идёт; см.
    # par:ssn-rpast). Данные ||B_k S_p - Y_p||_F для SR1 всё равно
    # сохранены в npz — на случай будущего реанализа.
    rpast_specs = [('psb', 'ss_psb', 'fig_ndim_stat_rpast_psb.pdf')]
    for base, ss, fname in rpast_specs:
        out_path = os.path.join(THESIS_DIR, fname)
        make_pair_figure_rpast(raw, base, ss, ecv_configs, mgh_problems_n10,
                               n_dirs, max_iter, out_path)

    # ============================================================
    #   Сырые + сводка
    # ============================================================
    save_dict = dict(n_dirs=n_dirs, max_iter=max_iter, tol=tol,
                     seed=20260503,
                     R0_table=str(R0_table))
    for nv in needed_ns:
        save_dict[f"U_n{nv}"] = U_by_n[nv]
    summary_lines = [
        f"# diag_ndim_stat.py — сводка success-rate / медианы",
        f"# n_dirs={n_dirs}  max_iter={max_iter}  tol={tol}  seed=20260503",
        "",
        f"{'problem':22s} {'n':>3s} {'method':8s} {'conv':>5s} "
        f"{'med_iter':>9s} {'med_∇f':>8s} {'R0':>8s}",
        "-"*72,
    ]
    for key, trajs in raw.items():
        short, nv, mk = key
        if mk not in methods_all:
            continue
        n_conv = sum(1 for t in trajs if t['converged'])
        med_iter = (int(np.median([t['iters'] for t in trajs if t['converged']]))
                    if n_conv > 0 else -1)
        med_calls = (int(np.median([t['n_g']  for t in trajs if t['converged']]))
                     if n_conv > 0 else -1)
        if (short, nv) not in R0_table:
            # checkpoint без R0 — восстанавливаем из определений задач
            if short == 'ros_chain':
                R0_table[(short, nv)] = norm(rosenbrock_chained(nv)['x0'])
            elif short == 'ext_rosen':
                R0_table[(short, nv)] = norm(extended_rosenbrock(nv)['x0'])
            elif short == 'ecv':
                R0_table[(short, nv)] = norm(extended_curved_valley(n=nv)['x0'])
        R0 = R0_table[(short, nv)]
        summary_lines.append(
            f"{short:22s} {nv:>3d} {mk:8s} {n_conv:>2d}/{n_dirs} "
            f"{med_iter:>9d} {med_calls:>8d} {R0:>8.3f}")
        # save raw arrays (truncated to max_iter+1 to bound memory)
        G = aggregate_g(trajs, max_iter)
        Rp = aggregate_rpast(trajs, max_iter)
        save_dict[f"{short}_n{nv}_{mk}_gnorm"] = G
        save_dict[f"{short}_n{nv}_{mk}_rpast"] = Rp
        save_dict[f"{short}_n{nv}_{mk}_converged"] = np.array(
            [t['converged'] for t in trajs])
        save_dict[f"{short}_n{nv}_{mk}_iters"] = np.array(
            [t['iters'] for t in trajs])
        save_dict[f"{short}_n{nv}_{mk}_ng"]    = np.array(
            [t['n_g']   for t in trajs])

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    npz_out = os.path.join(SCRIPT_DIR, "ndim_stat.npz")
    np.savez_compressed(npz_out, **save_dict)
    print(f"\nsaved raw: {npz_out}")
    summary_out = os.path.join(SCRIPT_DIR, "ndim_stat_summary.txt")
    with open(summary_out, "w") as fp:
        fp.write(summary + "\n")
    print(f"saved summary: {summary_out}")

    # finished — try to remove checkpoint (sandbox может запрещать unlink)
    try:
        if os.path.exists(CHECKPOINT):
            os.remove(CHECKPOINT)
    except OSError:
        pass


if __name__ == "__main__":
    main()
