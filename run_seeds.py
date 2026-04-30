"""
Мульти-seed эксперимент гл. 4 диссертации SP-Broyden.

Сильно монотонная VI с кубической нелинейностью:
    F(x) = A x + eps * (x - c)^{odot 3},   A = U diag(1,...,kappa) U^T,
где U случайная ортогональная, c ~ N(0, 0.09).

Прогоняются три метода (тот же набор, что в \\S sec:viji:numerics):
    * VIQA-Broyden:           v_k = s_k
    * VIJI-Restart + SP-Broyden:  v_k = S_p (S_p^T S_p)^{-1} e_1, окно p<=p_max
    * VIJI-Restart + SP-AFD:  то же + одна FD-поправка вдоль d_k^{(0)}

Схема итерации (упрощённый VIJI-Restart):
    x_{k+1} = x_k - (B_k + beta_k I)^{-1} F(x_k)
    beta_k  = max(0.1, L_1 * ||d_{k-1}||)
    рестарт секущей истории каждые N=25 итераций.

Output: results.npz, fig_viji_seeds.pdf
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------- Задача ----------------------------------------------------------
def make_problem(n: int, kappa: float, eps: float, seed: int):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(H)
    diag = np.linspace(1.0, kappa, n)
    A = Q @ np.diag(diag) @ Q.T
    c = rng.standard_normal(n) * 0.3       # std = 0.3, var = 0.09 (как в гл. 4)
    x_star = rng.standard_normal(n) * 0.5  # цель: строим F так, чтобы x* был корнем
    # Сдвинем c так, чтобы F(x*) = 0:  A x* + eps (x* - c)^3 = 0
    # → c = x* - cbrt(-A x* / eps)
    Ax = A @ x_star
    c = x_star - np.cbrt(-Ax / eps)
    L1 = 6.0 * eps                           # как в гл. 4

    def F(x):
        return A @ x + eps * (x - c) ** 3

    def J(x):
        return A + np.diag(3 * eps * (x - c) ** 2)

    return F, J, A, c, x_star, L1


# ---------- VIJI-Restart обёртка --------------------------------------------
def run_method(F, x0, x_star, *, method: str, p_max: int, beta_floor: float,
               L1: float, max_iter: int, restart_every: int, tol: float,
               cond_thresh: float, fd_h: float):
    """
    method ∈ {'broyden', 'sp', 'sp_afd'}.
    Возвращает массив ||x_k - x*|| длины <= max_iter+1 и счётчик F-вызовов.
    """
    n = x0.size
    x = x0.copy()
    B = np.eye(n)
    Sp = np.zeros((n, 0))     # секущие шаги
    Yp = np.zeros((n, 0))
    err = [np.linalg.norm(x - x_star)]
    fcount = [0]

    f_curr = F(x); fcount[-1] += 1
    d_prev = np.zeros(n)

    for k in range(max_iter):
        # шаг с регуляризацией
        beta_k = max(beta_floor, L1 * np.linalg.norm(d_prev))
        try:
            d = -np.linalg.solve(B + beta_k * np.eye(n), f_curr)
        except np.linalg.LinAlgError:
            break

        # Armijo backtracking на merit ψ = 1/2 ||F||^2 (только на первых шагах)
        psi = 0.5 * np.dot(f_curr, f_curr)
        alpha = 1.0
        for _ in range(20):
            x_try = x + alpha * d
            f_try = F(x_try)
            if not np.all(np.isfinite(f_try)):
                alpha *= 0.5; continue
            if 0.5 * np.dot(f_try, f_try) <= psi - 1e-4 * alpha * np.dot(f_curr, f_curr):
                break
            alpha *= 0.5
        d = alpha * d
        x_new = x + d
        f_new = F(x_new); fcount.append(fcount[-1] + 1)
        s = x_new - x
        y = f_new - f_curr

        # секущее обновление
        if method == 'broyden':
            v = s / max(np.dot(s, s), 1e-30)
            B = B + np.outer(y - B @ s, v)
        else:  # 'sp' или 'sp_afd'
            # добавляем (s, y) в окно
            Sp = np.column_stack([Sp, s])[:, -p_max:] if Sp.size else s[:, None]
            Yp = np.column_stack([Yp, y])[:, -p_max:] if Yp.size else y[:, None]
            # адаптивный sliding window: режем колонки слева, пока cond > thresh
            while Sp.shape[1] > 1:
                G = Sp.T @ Sp
                if np.linalg.cond(G) < cond_thresh:
                    break
                Sp = Sp[:, 1:]; Yp = Yp[:, 1:]
            # v = Sp (Sp^T Sp)^{-1} e_1, e_1 — вектор, отвечающий новейшему s_k (последний столбец)
            G = Sp.T @ Sp
            try:
                e_last = np.zeros(Sp.shape[1]); e_last[-1] = 1.0
                v = Sp @ np.linalg.solve(G, e_last)
            except np.linalg.LinAlgError:
                v = s / max(np.dot(s, s), 1e-30)
            B = B + np.outer(y - B @ s, v)
            # SP-AFD — ровно одна FD-поправка вдоль d_k^{(0)}
            if method == 'sp_afd':
                h = fd_h * max(1.0, np.linalg.norm(x_new))
                u = d / max(np.linalg.norm(d), 1e-30)
                f_plus = F(x_new + h * u); fcount[-1] += 1
                y_fd = (f_plus - f_new) / h          # ≈ J(x_new) u
                # обновим B на дополнительной паре (u, y_fd) тем же SP-правилом
                Sp_fd = np.column_stack([Sp, u])[:, -p_max:]
                Yp_fd = np.column_stack([Yp, y_fd])[:, -p_max:]
                G2 = Sp_fd.T @ Sp_fd
                try:
                    e_last = np.zeros(Sp_fd.shape[1]); e_last[-1] = 1.0
                    v2 = Sp_fd @ np.linalg.solve(G2, e_last)
                    B = B + np.outer(y_fd - B @ u, v2)
                    Sp, Yp = Sp_fd, Yp_fd
                except np.linalg.LinAlgError:
                    pass

        # рестарт секущей истории
        if (k + 1) % restart_every == 0:
            Sp = np.zeros((n, 0))
            Yp = np.zeros((n, 0))

        x, f_curr, d_prev = x_new, f_new, d
        err.append(np.linalg.norm(x - x_star))
        if err[-1] < tol:
            break

    return np.array(err), np.array(fcount[:len(err)])


# ---------- Прогон ----------------------------------------------------------
def main():
    n, kappa, eps = 30, 20.0, 1.5
    SEEDS = list(range(10))
    results = {m: [] for m in ('broyden', 'sp', 'sp_afd')}
    fcalls  = {m: [] for m in ('broyden', 'sp', 'sp_afd')}

    for seed in SEEDS:
        F, J, A, c, x_star, L1 = make_problem(n, kappa, eps, seed)
        rng = np.random.default_rng(1000 + seed)
        x0 = x_star + rng.standard_normal(n) * 0.3   # ||x0-x*||_2 ~ 1.6 для n=30
        for m in results:
            err, fc = run_method(F, x0, x_star, method=m, p_max=3,
                                 beta_floor=0.1, L1=L1, max_iter=120,
                                 restart_every=25, tol=1e-12,
                                 cond_thresh=1e3, fd_h=1e-6)
            results[m].append(err)
            fcalls[m].append(fc)

    # выровняем длины (паддинг последним значением, чтобы median работал)
    def pad(seqs, max_len):
        out = []
        for s in seqs:
            tail = np.full(max_len - len(s), s[-1])
            out.append(np.concatenate([s, tail]))
        return np.stack(out)

    max_len_iter = max(max(len(s) for s in results[m]) for m in results)
    print(f"max_len_iter={max_len_iter}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    titles = {'broyden': 'VIQA-Broyden', 'sp': 'VIJI-Restart + SP-Broyden',
              'sp_afd': 'VIJI-Restart + SP-AFD'}
    colors = {'broyden': '#888888', 'sp': '#3060c0', 'sp_afd': '#d03030'}

    # --- Левая панель: по итерациям ---
    for m in ('broyden', 'sp', 'sp_afd'):
        E = pad(results[m], max_len_iter)
        med = np.median(E, axis=0)
        q25 = np.quantile(E, 0.25, axis=0)
        q75 = np.quantile(E, 0.75, axis=0)
        xs = np.arange(med.size)
        axes[0].fill_between(xs, q25, q75, color=colors[m], alpha=0.18)
        axes[0].semilogy(xs, med, color=colors[m], lw=1.8, label=titles[m])
    axes[0].set_xlabel('итерация $k$')
    axes[0].set_ylabel(r'$\|x_k - x^{*}\|_2$')
    axes[0].set_title(f'Сходимость по итерациям ({len(SEEDS)} seed\'ов)')
    axes[0].grid(alpha=0.25, which='both')
    axes[0].legend()

    # --- Правая панель: по числу F-вызовов (строим как f-vs-err для каждого seed, потом median по сетке) ---
    F_GRID = np.arange(1, max(max(fc[-1] for fc in fcalls[m]) for m in fcalls) + 1)
    for m in ('broyden', 'sp', 'sp_afd'):
        # для каждого seed интерполируем err как ступенчатую функцию F-вызовов
        rows = []
        for err, fc in zip(results[m], fcalls[m]):
            # fc монотонно возрастает; на каждом F-значении возьмём err последнего достигнутого шага
            curve = np.full(F_GRID.size, err[-1])
            for fc_k, e in zip(fc, err):
                curve[F_GRID >= fc_k] = e
            rows.append(curve)
        E = np.stack(rows)
        med = np.median(E, axis=0)
        q25 = np.quantile(E, 0.25, axis=0)
        q75 = np.quantile(E, 0.75, axis=0)
        axes[1].fill_between(F_GRID, q25, q75, color=colors[m], alpha=0.18)
        axes[1].semilogy(F_GRID, med, color=colors[m], lw=1.8, label=titles[m])
    axes[1].set_xlabel(r'число вызовов $F$')
    axes[1].set_ylabel(r'$\|x_k - x^{*}\|_2$')
    axes[1].set_title(f'Сходимость по числу F-вызовов ({len(SEEDS)} seed\'ов)')
    axes[1].grid(alpha=0.25, which='both')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig('fig_viji_seeds.pdf', bbox_inches='tight')
    fig.savefig('fig_viji_seeds.png', bbox_inches='tight', dpi=150)
    np.savez('results.npz',
             **{f'{m}_err_seed{i}': e for m in results for i, e in enumerate(results[m])},
             **{f'{m}_fc_seed{i}':  c for m in fcalls  for i, c in enumerate(fcalls[m])})
    print("saved fig_viji_seeds.{pdf,png} and results.npz")

    # сводная статистика для текста ch4
    print("\n=== Median итераций до ||x-x*||<1e-8, по 10 seeds ===")
    for m in ('broyden', 'sp', 'sp_afd'):
        iters_to_tol = []
        f_to_tol = []
        for err, fc in zip(results[m], fcalls[m]):
            mask = err < 1e-8
            if mask.any():
                k = np.argmax(mask)
                iters_to_tol.append(k)
                f_to_tol.append(fc[k])
            else:
                iters_to_tol.append(np.nan)
                f_to_tol.append(np.nan)
        print(f"  {titles[m]:30s}  iter median={np.nanmedian(iters_to_tol):6.1f}"
              f"   IQR=[{np.nanquantile(iters_to_tol,0.25):.0f},{np.nanquantile(iters_to_tol,0.75):.0f}]"
              f"   #F median={np.nanmedian(f_to_tol):6.1f}")


if __name__ == '__main__':
    main()
