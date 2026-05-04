"""
diag_qn_compare.py — расширенное сравнение SS-SR1 с классическими и
limited-memory квазиньютоновскими методами на тестах MGH (n>=10).

Закрывает пункт P1 чеклиста «SS-SR1: сравнить с BFGS, PSB, L-BFGS, DFP».

Реализованы:
  SR1     :  B_{k+1} = B_k + r r^T/(r^T s),    r = y - B_k s
  SS-SR1  :  SR1 + MS-коррекция в s_k^⊥ (см.\ diag_ss_sr1.py)
  BFGS    :  B_{k+1} = B_k + y y^T/(y^T s) - B_k s s^T B_k/(s^T B_k s)
  DFP     :  обновление H_k = B_k^{-1} (двойственное к BFGS):
             H_{k+1} = (I - s y^T/(y^T s)) H_k (I - y s^T/(y^T s)) + s s^T/(y^T s)
             Реализуем как обновление H, направление d = -H g.
  PSB     :  обновление Пауэлла (симметризованная Broyden):
             B_{k+1} = B_k + (r s^T + s r^T)/(s^T s)
                            - (r^T s)(s s^T)/(s^T s)^2,   r = y - B_k s
  L-BFGS  :  двух-петельная рекурсия Носедала (m=10), хранение
             списка (s_i, y_i), без явного формирования B/H.

Линейный поиск — общий backtracking-Armijo по f (c1=1e-4, alpha←alpha/2)
для всех методов; стартовая B0 = I (или H0 = I); критерий остановки
||g||_2 <= 1e-8 или max_iter = 500.

Тестовые задачи (как в diag_ss_sr1.py):
  Rosenbrock chained        (n = 10)
  Extended Rosenbrock       (n = 10, MGH #21)
  Extended Powell singular  (n = 12, MGH #22)
  Extended Curved Valley    (n = 10, alpha=100, beta=1.5)

Выходные фигуры (mipt_thesis_master/):
  fig_qn_compare_conv.pdf      — log10(||g||) vs k для всех 6 методов
  fig_qn_compare_summary.pdf   — bar-chart числа итераций и #∇f
  fig_qn_compare_rpast.pdf     — невязка прошлых секущих ||R_past||_F (p=5)

Сырые: qn_compare.npz.
"""
from __future__ import annotations

import os
import numpy as np
from numpy.linalg import norm, solve, eigh, LinAlgError
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.join(SCRIPT_DIR, "mipt_thesis_master")


# ============================================================
#   Тестовые задачи (синхронизированы с diag_ss_sr1.py)
# ============================================================

def extended_rosenbrock(n=10):
    assert n % 2 == 0
    def f(x):
        s = 0.0
        for i in range(0, n, 2):
            s += 100.0*(x[i+1]-x[i]**2)**2 + (1.0-x[i])**2
        return s
    def g(x):
        gx = np.zeros(n)
        for i in range(0, n, 2):
            gx[i]   = -400.0*x[i]*(x[i+1]-x[i]**2) - 2.0*(1.0-x[i])
            gx[i+1] = 200.0*(x[i+1]-x[i]**2)
        return gx
    x0 = np.empty(n); x0[0::2] = -1.2; x0[1::2] = 1.0
    return dict(name="Extended Rosenbrock", n=n, f=f, g=g, x0=x0, fstar=0.0)


def extended_powell(n=12):
    assert n % 4 == 0
    def f(x):
        s = 0.0
        for j in range(0, n, 4):
            a = x[j]+10.0*x[j+1]; b = x[j+2]-x[j+3]
            c = x[j+1]-2.0*x[j+2]; d = x[j]-x[j+3]
            s += a*a + 5.0*b*b + c**4 + 10.0*d**4
        return s
    def g(x):
        gx = np.zeros(n)
        for j in range(0, n, 4):
            a = x[j]+10.0*x[j+1]; b = x[j+2]-x[j+3]
            c = x[j+1]-2.0*x[j+2]; d = x[j]-x[j+3]
            gx[j]   = 2.0*a + 40.0*d**3
            gx[j+1] = 20.0*a + 4.0*c**3
            gx[j+2] = 10.0*b - 8.0*c**3
            gx[j+3] = -10.0*b - 40.0*d**3
        return gx
    x0 = np.empty(n); pat = np.array([3.0, -1.0, 0.0, 1.0])
    for j in range(0, n, 4): x0[j:j+4] = pat
    return dict(name="Extended Powell singular", n=n, f=f, g=g, x0=x0, fstar=0.0)


def rosenbrock_chained(n=10):
    def f(x):
        s = 0.0
        for i in range(n-1):
            s += 100.0*(x[i+1]-x[i]**2)**2 + (1.0-x[i])**2
        return s
    def g(x):
        gx = np.zeros(n)
        for i in range(n-1):
            t = x[i+1]-x[i]**2
            gx[i]   += -400.0*x[i]*t - 2.0*(1.0-x[i])
            gx[i+1] += 200.0*t
        return gx
    x0 = np.empty(n); x0[0::2] = -1.2; x0[1::2] = 1.0
    return dict(name="Rosenbrock chained", n=n, f=f, g=g, x0=x0, fstar=0.0)


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
    return dict(name=f"Extended Curved Valley (a={alpha:.0f}, b={beta:g})",
                n=n, f=f, g=g, x0=x0, fstar=0.0)


# ============================================================
#   Линейный поиск
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
#   Обновления (для B-формы — Hessian approximation)
# ============================================================

def sr1_step(B, s, y, eps_skip=1e-8):
    r = y - B @ s
    den = float(r @ s)
    if abs(den) < eps_skip * max(norm(r)*norm(s), 1e-30):
        return B
    return B + np.outer(r, r) / den


def bfgs_step(B, s, y, eps_skip=1e-8):
    sy = float(s @ y)
    if sy <= eps_skip * max(norm(s)*norm(y), 1e-30):
        return B
    Bs = B @ s
    sBs = float(s @ Bs)
    if sBs <= 0:
        return B
    return B + np.outer(y, y)/sy - np.outer(Bs, Bs)/sBs


def psb_step(B, s, y):
    """Powell-Symmetric-Broyden: симметризованное Broyden-обновление.
    B_{k+1} = B_k + (r s^T + s r^T)/(s^T s) - (r^T s)(s s^T)/(s^T s)^2.
    Сохраняет симметрию, не требует кривизны s^T y > 0.
    Ref: Dennis & Schnabel (1996), §9.2.
    """
    r = y - B @ s
    ss = float(s @ s)
    if ss < 1e-30:
        return B
    rs = float(r @ s)
    return B + (np.outer(r, s) + np.outer(s, r))/ss - rs*np.outer(s, s)/(ss*ss)


def dfp_H_step(H, s, y, eps_skip=1e-8):
    """DFP-обновление обратной матрицы H = B^{-1}.
    H_{k+1} = (I - s y^T/sy) H_k (I - y s^T/sy) + s s^T/sy.
    """
    sy = float(s @ y)
    if sy <= eps_skip * max(norm(s)*norm(y), 1e-30):
        return H
    n = H.shape[0]
    I = np.eye(n)
    A = I - np.outer(s, y)/sy
    return A @ H @ A.T + np.outer(s, s)/sy


# ============================================================
#   L-BFGS: двух-петельная рекурсия
# ============================================================

def lbfgs_two_loop(g, s_list, y_list, gamma):
    """Возвращает d = -H_k g без формирования H_k.

    s_list, y_list — окно последних m пар, упорядоченных по возрастанию k
    (последний — самый новый). gamma = масштаб H0 = gamma I.
    """
    q = g.copy()
    m = len(s_list)
    rho = []
    alpha = [0.0]*m
    for i in range(m):
        sy = float(s_list[i] @ y_list[i])
        rho.append(1.0/sy if abs(sy) > 1e-30 else 0.0)
    # First loop (от нового к старому)
    for i in range(m-1, -1, -1):
        if rho[i] == 0.0:
            continue
        alpha[i] = rho[i] * float(s_list[i] @ q)
        q = q - alpha[i] * y_list[i]
    r = gamma * q
    # Second loop (от старого к новому)
    for i in range(m):
        if rho[i] == 0.0:
            continue
        beta = rho[i] * float(y_list[i] @ r)
        r = r + (alpha[i] - beta) * s_list[i]
    return -r  # d = -H g


# ============================================================
#   SS-SR1 (как в diag_ss_sr1.py)
# ============================================================

def ss_sr1_step(B, s, y, S_window, Y_window, eps_skip=1e-8):
    n = B.shape[0]
    Bp = sr1_step(B, s, y, eps_skip)
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


# ============================================================
#   Общий solver
# ============================================================

def run(prob, method, p_window=5, lbfgs_m=10, max_iter=500, tol=1e-8):
    n = prob['n']
    f, gfun, x0 = prob['f'], prob['g'], prob['x0']
    x = x0.copy()
    # Состояние, специфичное для метода:
    #   B-методы (sr1, ss_sr1, bfgs, psb): хранят B, считают d = -B^{-1} g.
    #   H-метод (dfp): хранит H ≈ (∇²f)^{-1}, считает d = -H g.
    #   L-BFGS: хранит окно (s,y), использует двух-петельную рекурсию.
    if method == 'dfp':
        H = np.eye(n)
        B = None
    elif method == 'lbfgs':
        s_hist = []  # списки np.ndarray
        y_hist = []
        B = None
    else:
        B = np.eye(n)
    g = gfun(x); fx = f(x)
    n_f, n_g = 1, 1
    hist_g = [norm(g)]
    hist_Rpast = [0.0]
    S_buf = np.zeros((n, p_window))
    Y_buf = np.zeros((n, p_window))
    m_buf = 0
    converged = False
    status = "max_iter"
    for k in range(max_iter):
        if hist_g[-1] <= tol:
            converged = True; status = "converged"; break
        # ---- direction ----
        if method == 'dfp':
            d = -H @ g
        elif method == 'lbfgs':
            if len(s_hist) > 0:
                # gamma_k = (s_{k-1}·y_{k-1}) / (y_{k-1}·y_{k-1}) — стандарт
                ss, yy = s_hist[-1], y_hist[-1]
                yy2 = float(yy @ yy)
                gamma = float(ss @ yy)/yy2 if yy2 > 1e-30 else 1.0
            else:
                gamma = 1.0
            d = lbfgs_two_loop(g, s_hist, y_hist, gamma)
        else:
            try:
                d = solve(B, -g)
            except LinAlgError:
                d = -g
            if not np.all(np.isfinite(d)) or g @ d >= 0:
                d = -g
        # ---- line search ----
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
        g_new = gfun(x_new); n_g += 1
        f_new = f(x_new);    n_f += 1
        y = g_new - g
        # окно для R_past диагностики (одинаково для всех методов)
        if m_buf < p_window:
            S_buf[:, m_buf] = s; Y_buf[:, m_buf] = y; m_buf += 1
        else:
            S_buf[:, :-1] = S_buf[:, 1:]; S_buf[:, -1] = s
            Y_buf[:, :-1] = Y_buf[:, 1:]; Y_buf[:, -1] = y
        # ---- update ----
        if method == 'sr1':
            B = sr1_step(B, s, y)
        elif method == 'bfgs':
            B = bfgs_step(B, s, y)
        elif method == 'psb':
            B = psb_step(B, s, y)
        elif method == 'ss_sr1':
            B = ss_sr1_step(B, s, y, S_buf[:, :m_buf], Y_buf[:, :m_buf])
        elif method == 'dfp':
            H = dfp_H_step(H, s, y)
        elif method == 'lbfgs':
            s_hist.append(s.copy()); y_hist.append(y.copy())
            if len(s_hist) > lbfgs_m:
                s_hist.pop(0); y_hist.pop(0)
        else:
            raise ValueError(method)
        # R_past: для DFP / L-BFGS вычислим от обращения, чтобы метрика
        # «прошлые секущие» имела смысл. Для DFP B = H^{-1}; для L-BFGS
        # формирование B нерационально — используем surrogate ||S^T H y - S^T s||
        if m_buf >= 1:
            if method == 'dfp':
                B_eff = solve(H, np.eye(n))  # n<=20, цена пренебрежима
                R = B_eff @ S_buf[:, :m_buf] - Y_buf[:, :m_buf]
            elif method == 'lbfgs':
                # surrogate: ||H y_i - s_i||  усреднён по окну
                ss, yy = s_hist[-1], y_hist[-1]
                yy2 = float(yy @ yy)
                gamma = float(ss @ yy)/yy2 if yy2 > 1e-30 else 1.0
                # Применяем H_k к каждому y_j из окна и сравниваем с s_j
                R_cols = np.zeros((n, m_buf))
                for j in range(m_buf):
                    # Поскольку H = -d/g, оценим через две-петельную:
                    # h_j = - lbfgs_two_loop(-Y[:,j], s_hist, y_hist, gamma) дает H y_j
                    h_j = - lbfgs_two_loop(-Y_buf[:, j], s_hist, y_hist, gamma)
                    R_cols[:, j] = h_j - S_buf[:, j]
                R = R_cols  # размерности (n,m); это «обратная» невязка
            else:
                R = B @ S_buf[:, :m_buf] - Y_buf[:, :m_buf]
            hist_Rpast.append(float(np.linalg.norm(R, 'fro')))
        else:
            hist_Rpast.append(0.0)
        x, g, fx = x_new, g_new, f_new
        hist_g.append(norm(g))
    return dict(method=method, problem=prob['name'], n=n,
                converged=converged, status=status,
                iters=len(hist_g)-1, n_f=n_f, n_g=n_g,
                hist_g=np.array(hist_g),
                hist_Rpast=np.array(hist_Rpast),
                g_final_norm=norm(g))


# ============================================================
#   main
# ============================================================

def main():
    rng = np.random.default_rng(20260428)  # consistent with diag_ss_sr1.py
    problems = [rosenbrock_chained(10),
                extended_rosenbrock(10),
                extended_powell(12),
                extended_curved_valley(n=10, alpha=100.0, beta=1.5)]
    methods = [('sr1',    'SR1',    'tab:blue',   '-'),
               ('ss_sr1', 'SS-SR1', 'tab:red',    '-'),
               ('bfgs',   'BFGS',   'tab:green',  '--'),
               ('dfp',    'DFP',    'tab:purple', '-.'),
               ('psb',    'PSB',    'tab:orange', ':'),
               ('lbfgs',  'L-BFGS (m=10)', 'tab:brown', '--')]
    results = {}
    for prob in problems:
        for m, *_ in methods:
            r = run(prob, m, p_window=5, lbfgs_m=10, max_iter=500, tol=1e-8)
            results[(prob['name'], m)] = r
            print(f"{prob['name']:32s} {m:6s} conv={str(r['converged']):5s} "
                  f"iters={r['iters']:4d}  nf={r['n_f']:5d}  ||g||={r['g_final_norm']:.2e}")

    # ---- Figure 1: convergence ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flat
    for ax, prob in zip(axes, problems):
        for m, lab, col, ls in methods:
            r = results[(prob['name'], m)]
            ks = np.arange(len(r['hist_g']))
            ax.semilogy(ks, np.maximum(r['hist_g'], 1e-16),
                        lw=1.5, ls=ls, color=col,
                        label=f"{lab} ({r['iters']}, {r['status']})")
        ax.set_title(f"{prob['name']} ($n={prob['n']}$)", fontsize=11)
        ax.set_xlabel(r"итерация $k$")
        ax.set_ylabel(r"$\|\nabla f(x_k)\|_2$")
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=7, loc='upper right')
        ax.axhline(1e-8, color='gray', ls=':', lw=0.8)
    fig.suptitle(r"SS-SR1 vs SR1, BFGS, DFP, PSB, L-BFGS на тестах MGH ($n\geq 10$)",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    out1 = os.path.join(THESIS_DIR, "fig_qn_compare_conv.pdf")
    fig.savefig(out1, bbox_inches='tight'); plt.close(fig)
    print(f"saved: {out1}")

    # ---- Figure 2: summary bars ----
    pnames = [p['name'] for p in problems]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bw = 0.13
    xs = np.arange(len(pnames))
    for ai, key in enumerate(['iters', 'n_g']):
        ax = axes[ai]
        for mi, (mk, mlab, mcol, _) in enumerate(methods):
            vals = []
            for p in problems:
                r = results[(p['name'], mk)]
                v = r[key] if r['converged'] else np.nan
                vals.append(v)
            offs = (mi - len(methods)/2 + 0.5)*bw
            ax.bar(xs + offs, vals, bw, color=mcol, label=mlab,
                   edgecolor='black', linewidth=0.4)
            for xi, v in zip(xs + offs, vals):
                if np.isnan(v):
                    ax.text(xi, 1, '—', ha='center', va='bottom',
                            fontsize=8, color='red')
                else:
                    ax.text(xi, v, f'{int(v)}', ha='center', va='bottom',
                            fontsize=7, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([p.replace(' ','\n', 1) for p in pnames],
                           fontsize=9)
        ax.set_ylabel({'iters': r'Итераций до $\|g\|\leq 10^{-8}$',
                       'n_g':   r'$\#\nabla f$ до сходимости'}[key])
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8, ncol=3, loc='upper left')
    fig.suptitle("Стоимость сходимости: 6 квазиньютоновских методов на 4 задачах",
                 fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    out2 = os.path.join(THESIS_DIR, "fig_qn_compare_summary.pdf")
    fig.savefig(out2, bbox_inches='tight'); plt.close(fig)
    print(f"saved: {out2}")

    # ---- Figure 3: ||R_past||_F (только для B-методов) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flat
    sel = [m for m in methods if m[0] not in ('dfp', 'lbfgs')]
    for ax, prob in zip(axes, problems):
        for m, lab, col, ls in sel:
            r = results[(prob['name'], m)]
            ks = np.arange(len(r['hist_Rpast']))
            ax.semilogy(ks, np.maximum(r['hist_Rpast'], 1e-16),
                        lw=1.5, ls=ls, color=col, label=lab)
        ax.set_title(f"{prob['name']} ($n={prob['n']}$)", fontsize=11)
        ax.set_xlabel(r"итерация $k$")
        ax.set_ylabel(r"$\|B_k S_p - Y_p\|_F$  (окно $p=5$)")
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(r"Невязка прошлых секущих (B-методы): "
                 r"только SS-SR1 минимизирует её по построению", fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.97])
    out3 = os.path.join(THESIS_DIR, "fig_qn_compare_rpast.pdf")
    fig.savefig(out3, bbox_inches='tight'); plt.close(fig)
    print(f"saved: {out3}")

    # ---- raw ----
    npz_out = os.path.join(SCRIPT_DIR, "qn_compare.npz")
    save_data = {}
    for (pname, mk), r in results.items():
        prefix = f"{pname.replace(' ','_').replace('(','').replace(')','').replace('=','').replace(',','').replace('.','p')}__{mk}"
        save_data[prefix + "__hist_g"]    = r['hist_g']
        save_data[prefix + "__hist_Rpast"] = r['hist_Rpast']
        save_data[prefix + "__iters"]     = r['iters']
        save_data[prefix + "__n_f"]       = r['n_f']
        save_data[prefix + "__n_g"]       = r['n_g']
        save_data[prefix + "__conv"]      = int(r['converged'])
    np.savez_compressed(npz_out, **save_data)
    print(f"saved raw: {npz_out}")

    # ---- summary table ----
    print("\n=== Summary ===")
    print(f"{'Problem':30s} {'method':8s} {'conv':5s} {'iters':>6s} {'#g':>5s} {'#f':>6s}")
    for prob in problems:
        for m, *_ in methods:
            r = results[(prob['name'], m)]
            print(f"{prob['name']:30s} {m:8s} {str(r['converged']):5s} "
                  f"{r['iters']:6d} {r['n_g']:5d} {r['n_f']:6d}")


if __name__ == "__main__":
    main()
