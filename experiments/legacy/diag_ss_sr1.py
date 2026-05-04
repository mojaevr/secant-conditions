"""
diag_ss_sr1.py — n-мерная диагностика SS-SR1 на тестовых задачах
Moré–Garbow–Hillstrom (n>=10).

Закрывает пункт P1 чеклиста «SS-SR1: все эксперименты в n=2».

Реализация SS-SR1 общая (не привязана к n=2):
  Шаг 1.  B'_k = B_k + (r r^T) / (r^T s_k),  r = y_k - B_k s_k          (SR1)
  Шаг 2.  Накопить окно p прошлых пар (s_j, y_j),  j = max(0, k-p+1)..k
          Построить S_p ∈ R^{n x m}, Y_p ∈ R^{n x m},  m = min(p, k+1).
          R'_k = B'_k S_p - Y_p,
          M'_k = 0.5 * (R'_k S_p^T + S_p R'_k^T),
          P_k  = I - ŝ_k ŝ_k^T,
          A_k  = P_k M'_k P_k,
          u_k* = ведущий собственный вектор A_k (по |λ|),
          σ_k* = -(u^T R'_k S_p^T u) / ||S_p^T u||^2,
  Шаг 3.  B_{k+1} = B'_k + σ_k* u_k* u_k*^T.

Тестовые задачи (MGH + extended Curved Valley):
  - Rosenbrock chained        (n = 10)
  - Extended Rosenbrock       (n = 10)
  - Extended Powell singular  (n = 12)
  - Extended Curved Valley    (n = 10, α=50, β=1)
       — векторизация задачи из basin.ipynb (n=2). Имеет вращение
         главных осей гессиана вдоль траектории, на котором классический
         SR1 деградирует. Используется как stress-test, на котором
         MS-коррекция в s⊥ должна давать выигрыш.

Сравниваем: SR1 vs SS-SR1 (основной фокус); BFGS добавлен как
референсный baseline для контекста.

Все методы используют один и тот же Armijo-backtracking ($c_1 = 10^{-4}$,
откат $\alpha \leftarrow \alpha/2$) и один и тот же критерий остановки
||g||_2 <= eps_tol = 1e-8 или max_iter = 500.

Выходные фигуры:
  fig_ss_sr1_ndim_conv.pdf     — сходимость на 4 задачах, log10(||g||) vs k
  fig_ss_sr1_ndim_summary.pdf  — bar-chart числа итераций и #∇f до tol

Сырые результаты в ss_sr1_ndim.npz.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, solve, eigh, LinAlgError
import matplotlib.pyplot as plt

# ============================================================
#   Тестовые задачи MGH
# ============================================================

def extended_rosenbrock(n=10):
    """f = sum_{i=1..n/2} [100(x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2],
    x* = (1,...,1), x0 = (-1.2, 1, -1.2, 1, ...)."""
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
    x0 = np.empty(n)
    x0[0::2] = -1.2
    x0[1::2] = 1.0
    return dict(name="Extended Rosenbrock", n=n, f=f, g=g, x0=x0,
                fstar=0.0)


def extended_powell(n=12):
    """f = sum_{i=1..n/4} [
            (x_{4i-3} + 10 x_{4i-2})^2
          + 5(x_{4i-1} - x_{4i})^2
          + (x_{4i-2} - 2 x_{4i-1})^4
          + 10(x_{4i-3} - x_{4i})^4 ],
    x* = 0, x0 = (3, -1, 0, 1, 3, -1, 0, 1, ...)."""
    assert n % 4 == 0
    def f(x):
        s = 0.0
        for j in range(0, n, 4):
            a = x[j]   + 10.0*x[j+1]
            b = x[j+2] -      x[j+3]
            c = x[j+1] - 2.0 *x[j+2]
            d = x[j]   -      x[j+3]
            s += a*a + 5.0*b*b + c**4 + 10.0*d**4
        return s
    def g(x):
        gx = np.zeros(n)
        for j in range(0, n, 4):
            a = x[j]   + 10.0*x[j+1]
            b = x[j+2] -      x[j+3]
            c = x[j+1] - 2.0 *x[j+2]
            d = x[j]   -      x[j+3]
            gx[j]   = 2.0*a + 40.0*d**3
            gx[j+1] = 20.0*a + 4.0*c**3
            gx[j+2] = 10.0*b - 8.0*c**3
            gx[j+3] = -10.0*b - 40.0*d**3
        return gx
    x0 = np.empty(n)
    pat = np.array([3.0, -1.0, 0.0, 1.0])
    for j in range(0, n, 4):
        x0[j:j+4] = pat
    return dict(name="Extended Powell singular", n=n, f=f, g=g, x0=x0,
                fstar=0.0)


def rosenbrock_chained(n=10):
    """Цепной Розенброк (Nocedal–Wright form):
    f = sum_{i=1..n-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2],
    x* = (1,...,1), x0 = (-1.2, 1, -1.2, 1, ...)."""
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
    x0 = np.empty(n)
    x0[0::2] = -1.2; x0[1::2] = 1.0
    return dict(name="Rosenbrock chained", n=n, f=f, g=g, x0=x0,
                fstar=0.0)


def extended_curved_valley(n=10, alpha=50.0, beta=1.0):
    """Векторизация задачи из basin.ipynb на n=2 в n переменных.

    f(x) = sum_{i=1..n/2} [α/2 (x_{2i} - β x_{2i-1}^2)^2
                          + x_{2i-1}^2/2 + x_{2i}^2/2],
    x* = 0, J* = block_diag( diag(1, 1+α) ) — каждый блок 2x2.
    Особенность: вдоль траектории главные оси гессиана вращаются
    при β > 0, что было базовым stress-test'ом для SS-SR1 при n=2.
    Здесь n>=10, что выводит задачу из 2D-частного случая.
    """
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
    # Старт: смещение из x* по 2-блокам, амплитуда 1.8 (тот же режим,
    # что в basin.ipynb: x0=(1.8,1.8) при α=100, β=1.5).
    x0 = np.empty(n)
    x0[0::2] = 1.8
    x0[1::2] = 1.8
    return dict(name=f"Extended Curved Valley (α={alpha:.0f}, β={beta:g})",
                n=n, f=f, g=g, x0=x0, fstar=0.0)


# ============================================================
#   Линейный поиск и общий solver
# ============================================================

def armijo_f(f, x, fx, g, d, c1=1e-4, alpha0=1.0, max_bt=40):
    """Стандартный backtracking-Armijo на f:
        f(x + α d) ≤ f(x) + c1 α (g, d).
    Используется одинаково для SR1, SS-SR1, BFGS, чтобы сравнение
    между методами зависело только от выбора обновления B_k.

    Возвращает (alpha, n_f).
    """
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
#   Квазиньютоновские обновления
# ============================================================

def sr1_step(B, s, y, eps_skip=1e-8):
    r = y - B @ s
    den = float(r @ s)
    nr  = norm(r); ns = norm(s)
    if abs(den) < eps_skip * max(nr*ns, 1e-30):
        return B, False  # skip — секущая ослабленная
    return B + np.outer(r, r) / den, True


def bfgs_step(B, s, y, eps_skip=1e-8):
    sy = float(s @ y)
    if sy <= eps_skip * max(norm(s)*norm(y), 1e-30):
        return B, False
    Bs = B @ s
    sBs = float(s @ Bs)
    if sBs <= 0:
        return B, False
    B = B + np.outer(y, y)/sy - np.outer(Bs, Bs)/sBs
    return B, True


def ss_sr1_step(B, s, y, S_window, Y_window, eps_skip=1e-8):
    """Полная n-мерная SS-SR1.

    S_window, Y_window — массивы np.ndarray (n, m), m = текущий размер окна,
    включают (s, y) текущей итерации (т.е. m >= 1, и последний столбец
    = s, y).
    """
    n = B.shape[0]
    # Шаг 1. SR1
    Bp, _ = sr1_step(B, s, y, eps_skip=eps_skip)
    # Шаг 2. MS-коррекция
    if S_window is None or S_window.shape[1] == 0:
        return Bp, "sr1_only"
    R = Bp @ S_window - Y_window                    # (n, m)
    M = 0.5*(R @ S_window.T + S_window @ R.T)        # (n, n) симметричная
    s_hat = s / max(norm(s), 1e-30)
    # P @ M @ P через явный проектор; n у нас 10..20, цена пренебрежима
    P = np.eye(n) - np.outer(s_hat, s_hat)
    A = P @ M @ P
    A = 0.5*(A + A.T)  # симметризация против ошибок плавающего
    try:
        w, V = eigh(A)
    except LinAlgError:
        return Bp, "eig_fail"
    # ведущий по |λ| (как в опр. 3.X — оптимизация по знаку учтена в σ*)
    j = int(np.argmax(np.abs(w)))
    u = V[:, j]
    # численная нормировка и проекция в s⊥ (страховка)
    u = u - (u @ s_hat)*s_hat
    nu = norm(u)
    if nu < 1e-12:
        return Bp, "u_degenerate"
    u /= nu
    Stu = S_window.T @ u                              # (m,)
    den = float(Stu @ Stu)
    if den < 1e-15:
        return Bp, "Stu_zero"
    sigma = -float(u @ (R @ Stu)) / den
    return Bp + sigma*np.outer(u, u), "ok"


# ============================================================
#   Общий solver
# ============================================================

def run(prob, method, p_window=5, max_iter=500, tol=1e-8, verbose=False):
    """Возвращает dict со статистикой и траекторией.

    Линейный поиск — Armijo на f (стандартный для минимизации). Все
    три метода используют ОДИН И ТОТ ЖЕ глобализатор, поэтому различия
    результатов отражают только выбор обновления B_k.

    Отдельно отслеживается ||R_past||_F = ||B_k S_p - Y_p||_F по фикс.
    окну p прошлых секущих — прямая проверка Th. 3.X (монотонность
    SS-SR1) против SR1, который этот residual не контролирует.
    """
    n = prob['n']
    f, gfun, x0 = prob['f'], prob['g'], prob['x0']
    x = x0.copy()
    B = np.eye(n)
    g = gfun(x)
    fx = f(x)
    n_f, n_g = 1, 1
    hist_g = [norm(g)]
    hist_f = [fx]
    hist_Rpast = [0.0]
    S_buf = np.zeros((n, p_window))
    Y_buf = np.zeros((n, p_window))
    m_buf = 0
    converged = False
    status = "max_iter"
    for k in range(max_iter):
        if hist_g[-1] <= tol:
            converged = True
            status = "converged"
            break
        try:
            d = solve(B, -g)
        except LinAlgError:
            d = -g
        if not np.all(np.isfinite(d)):
            d = -g
        if g @ d >= 0:
            d = -g  # направление подъёма — fallback на градиент
        a, nf_ls = armijo_f(f, x, fx, g, d)
        n_f += nf_ls
        if a is None:
            d = -g
            a, nf_ls = armijo_f(f, x, fx, g, d)
            n_f += nf_ls
            if a is None:
                status = "ls_fail"
                break
        s = a*d
        x_new = x + s
        g_new = gfun(x_new); n_g += 1
        f_new = f(x_new);    n_f += 1
        y = g_new - g
        # обновляем окно (для всех методов — нужно для R_past диагностики)
        if m_buf < p_window:
            S_buf[:, m_buf] = s; Y_buf[:, m_buf] = y; m_buf += 1
        else:
            S_buf[:, :-1] = S_buf[:, 1:]; S_buf[:, -1] = s
            Y_buf[:, :-1] = Y_buf[:, 1:]; Y_buf[:, -1] = y
        # обновление B
        if method == 'sr1':
            B, _ = sr1_step(B, s, y)
        elif method == 'bfgs':
            B, _ = bfgs_step(B, s, y)
        elif method == 'ss_sr1':
            B, _ = ss_sr1_step(B, s, y, S_buf[:, :m_buf], Y_buf[:, :m_buf])
        else:
            raise ValueError(method)
        # ||R_past||_F: невязка ВСЕГО окна (включая текущую секущую)
        # после обновления — этот residual SS-SR1 минимизирует, SR1 — нет.
        if m_buf >= 1:
            R = B @ S_buf[:, :m_buf] - Y_buf[:, :m_buf]
            hist_Rpast.append(float(np.linalg.norm(R, 'fro')))
        else:
            hist_Rpast.append(0.0)
        x, g, fx = x_new, g_new, f_new
        hist_g.append(norm(g))
        hist_f.append(fx)
    return dict(method=method, problem=prob['name'], n=n,
                converged=converged, status=status,
                iters=len(hist_g)-1, n_f=n_f, n_g=n_g,
                hist_g=np.array(hist_g), hist_f=np.array(hist_f),
                hist_Rpast=np.array(hist_Rpast),
                x_final=x, g_final_norm=norm(g))


# ============================================================
#   main
# ============================================================

def main():
    rng = np.random.default_rng(20260428)  # seed-policy из reproducibility.tex
    problems = [rosenbrock_chained(10),
                extended_rosenbrock(10),
                extended_powell(12),
                extended_curved_valley(n=10, alpha=100.0, beta=1.5)]
    methods = [('sr1',    'SR1',    'tab:blue',   '-'),
               ('ss_sr1', 'SS-SR1', 'tab:red',    '-'),
               ('bfgs',   'BFGS',   'tab:green',  '--')]
    results = {}
    for prob in problems:
        for m, *_ in methods:
            r = run(prob, m, p_window=5, max_iter=500, tol=1e-8)
            key = (prob['name'], m)
            results[key] = r
            print(f"{prob['name']:30s}  {m:8s}  conv={r['converged']!s:5s} "
                  f"iters={r['iters']:4d}  nf={r['n_f']:5d} ng={r['n_g']:4d}  "
                  f"||g||={r['g_final_norm']:.2e}")
    # ---- Figure 1: convergence curves (||g||) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    axes = axes.flat
    for ax, prob in zip(axes, problems):
        for m, lab, col, ls in methods:
            r = results[(prob['name'], m)]
            ks = np.arange(len(r['hist_g']))
            ax.semilogy(ks, np.maximum(r['hist_g'], 1e-16),
                        lw=1.6, ls=ls, color=col,
                        label=f"{lab} ({r['iters']} ит., {r['status']})")
        ax.set_title(f"{prob['name']} ($n={prob['n']}$)", fontsize=11)
        ax.set_xlabel(r"итерация $k$")
        ax.set_ylabel(r"$\|\nabla f(x_k)\|_2$")
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
        ax.axhline(1e-8, color='gray', ls=':', lw=0.8)
    fig.suptitle(r"SS-SR1 vs SR1 vs BFGS: сходимость на тестах MGH+CV ($n \geq 10$)",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    out1 = "/sessions/eloquent-tender-faraday/mnt/SP-Broyden/mipt_thesis_master/fig_ss_sr1_ndim_conv.pdf"
    fig.savefig(out1, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out1}")
    # ---- Figure 1b: past-secant residual ||R_past||_F (проверка теории) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flat
    for ax, prob in zip(axes, problems):
        for m, lab, col, ls in methods:
            r = results[(prob['name'], m)]
            ks = np.arange(len(r['hist_Rpast']))
            ax.semilogy(ks, np.maximum(r['hist_Rpast'], 1e-16),
                        lw=1.6, ls=ls, color=col, label=lab)
        ax.set_title(f"{prob['name']} ($n={prob['n']}$)", fontsize=11)
        ax.set_xlabel(r"итерация $k$")
        ax.set_ylabel(r"$\|B_k S_p - Y_p\|_F$  (окно $p=5$)")
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(r"Невязка прошлых секущих $\|R_{\mathrm{past}}\|_F$: "
                 r"SS-SR1 минимизирует её по построению",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    out1b = "/sessions/eloquent-tender-faraday/mnt/SP-Broyden/mipt_thesis_master/fig_ss_sr1_ndim_rpast.pdf"
    fig.savefig(out1b, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out1b}")
    # ---- Figure 2: summary bars ----
    pnames = [p['name'] for p in problems]
    method_keys = [m[0] for m in methods]
    method_labs = [m[1] for m in methods]
    method_cols = [m[2] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bw = 0.26
    xs = np.arange(len(pnames))
    for ai, key in enumerate(['iters', 'n_g']):
        ax = axes[ai]
        for mi, (mk, mlab, mcol) in enumerate(zip(method_keys, method_labs, method_cols)):
            vals = []
            for p in problems:
                r = results[(p['name'], mk)]
                v = r[key] if r['converged'] else np.nan
                vals.append(v)
            ax.bar(xs + (mi-1)*bw, vals, bw, color=mcol, label=mlab,
                   edgecolor='black', linewidth=0.4)
            for xi, v in zip(xs + (mi-1)*bw, vals):
                if np.isnan(v):
                    ax.text(xi, 1, '—', ha='center', va='bottom',
                            fontsize=9, color='red')
                else:
                    ax.text(xi, v, f'{int(v)}', ha='center', va='bottom',
                            fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([p.replace(' ','\n', 1) for p in pnames],
                           fontsize=9)
        ax.set_ylabel({'iters': r'Итераций до $\|g\|\leq 10^{-8}$',
                       'n_g':   r'$\#\nabla f$ до сходимости'}[key])
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("SS-SR1 vs SR1 vs BFGS: цена сходимости на 4 задачах",
                 fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    out2 = "/sessions/eloquent-tender-faraday/mnt/SP-Broyden/mipt_thesis_master/fig_ss_sr1_ndim_summary.pdf"
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out2}")
    # ---- raw data ----
    npz_out = "/sessions/eloquent-tender-faraday/mnt/SP-Broyden/ss_sr1_ndim.npz"
    save_data = {}
    for (pname, mk), r in results.items():
        prefix = f"{pname.replace(' ','_')}__{mk}"
        save_data[prefix + "__hist_g"] = r['hist_g']
        save_data[prefix + "__hist_f"] = r['hist_f']
        save_data[prefix + "__hist_Rpast"] = r['hist_Rpast']
        save_data[prefix + "__iters"]  = r['iters']
        save_data[prefix + "__n_f"]    = r['n_f']
        save_data[prefix + "__n_g"]    = r['n_g']
        save_data[prefix + "__conv"]   = int(r['converged'])
    np.savez_compressed(npz_out, **save_data)
    print(f"saved raw: {npz_out}")
    # ---- summary table ----
    print("\n=== Summary table ===")
    print(f"{'Problem':32s} {'method':8s} {'conv':5s} {'iters':>5s} {'#g':>5s} {'#f':>6s} {'||g||':>10s}")
    for prob in problems:
        for m, *_ in methods:
            r = results[(prob['name'], m)]
            print(f"{prob['name']:32s} {m:8s} {str(r['converged']):5s} "
                  f"{r['iters']:5d} {r['n_g']:5d} {r['n_f']:6d} "
                  f"{r['g_final_norm']:10.2e}")


if __name__ == "__main__":
    main()
