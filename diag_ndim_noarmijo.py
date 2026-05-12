"""diag_ndim_noarmijo.py — библиотека solver'а БЕЗ Armijo.

Экспортирует `run_no_armijo`: полный QN-шаг $x_{k+1} = x_k + d_k$
(без backtracking, без fallback'а на $-\\nabla f$), $B_0=I$, шаг через
LU-разложение системы $B_k d_k = -\\nabla f(x_k)$. Используется
`diag_ss_local_basin.py` для бассейновой картинки.

Параметры обновления — те~же, что в `diag_ndim_stat.run`
(скрывают SS-коррекцию `_ss_correct`), отличие только в~отсутствии Armijo.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, solve, LinAlgError

from diag_ndim_stat import (
    sr1_step,
    psb_step,
    ss_sr1_step,
    ss_psb_step,
)


def run_no_armijo(prob, x0, method, p_window=5, max_iter=500, tol=1e-8):
    """Тот же контракт, что diag_ndim_stat.run, но без Armijo: a≡1, без fallback."""
    n = prob['n']
    f, gfun = prob['f'], prob['g']
    x = x0.copy()
    B = np.eye(n)

    g = gfun(x)
    n_f, n_g = 0, 1
    hist_g = [norm(g)]
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
            status = "ls_fail"; break  # сингулярный B без Armijo — труба

        if not np.all(np.isfinite(d)):
            status = "diverge"; break

        s = d  # полный QN-шаг
        x_new = x + s
        if not np.all(np.isfinite(x_new)) or norm(x_new) > 1e10:
            status = "diverge"; break

        g_new = gfun(x_new); n_g += 1
        if not np.all(np.isfinite(g_new)):
            status = "diverge"; break
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
        else:
            raise ValueError(method)

        if not np.all(np.isfinite(B)):
            status = "diverge"; break

        x, g = x_new, g_new
        hist_g.append(norm(g))

    return dict(method=method, problem=prob['short'], n=n,
                converged=converged, status=status,
                iters=len(hist_g)-1, n_f=n_f, n_g=n_g,
                g_final=hist_g[-1])


