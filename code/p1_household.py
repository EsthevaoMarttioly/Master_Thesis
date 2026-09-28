#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid, and transition matrices for income, labor productivity, and sectoral status.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sequence_jacobian import het, interpolate, grids


# ---------------------------------------------------------------------------
# 1. Utility, Grid, and Income

# 1.1. Utility Functions
nB, nS, nF = 2, 3, 2    # beta_grid x labor_grid x BF status size

expand = lambda x: np.tile(np.repeat(x[:, None, :], nB, 1).reshape(-1), nF)
zeros  = lambda x, n: [np.zeros_like(x) for _ in range(n)]
states = lambda x: [slice(k * x.shape[0] // nS, (k+1) * x.shape[0] // nS) for k in range(nS)]

u = lambda c, eis: np.log(np.maximum(c, 1e-12)) if eis == 1 else\
                            np.maximum(c, 1e-12)  ** (1-1/eis) / (1-1/eis)
v  = lambda h, psi, varphi: psi * np.maximum(h, 1e-8) ** (1+1/varphi)/(1+1/varphi)
uc = lambda c, eis: np.maximum(c, 1e-12) ** (-1/eis)


# 1.2. Exogenous Transition States Grid
def discretize_normal(mu, sigma, n):
    # theta_s ~ N(mu_s, sigma_s^2) as Equiprobable Bins.
    z = norm.ppf(np.arange(n + 1) / n)
    s = n * (norm.pdf(z[:-1]) - norm.pdf(z[1:]))
    theta = mu + sigma * s / np.sqrt(np.mean(s ** 2))
    prob  = np.full(n, 1.0 / n)
    return theta, prob


def make_egrid(rho_e, sd_e, nE, amin, amax, nA,
               sigma_F, mu_I, sigma_I, nT):
    # Productivity Grids
    e_grid, pi_e_e, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    e_grid        = e_grid / np.sum(pi_e_e * e_grid)
    thetaF, probF = discretize_normal(0.0, sigma_F, nT)
    thetaF        = thetaF - np.log(probF @ np.exp(thetaF))   # Normalized: E(exp(theta_F)) = 1
    thetaI, probI = discretize_normal(mu_I, sigma_I, nT)

    # Asset Grid
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi_e, a_grid, thetaF, probF, thetaI, probI


def make_bgrid(beta_high, dbeta, omega_I, q, nE, nT):
    # Build the beta grid for discount factors.
    beta_low = beta_high - dbeta
    b_grid   = np.array([beta_low, beta_high])
    pi_b = np.array([omega_I, 1-omega_I])
    Pi_b = (1 - q) * np.eye(nB) + q * np.outer(np.ones(nB), pi_b)
    beta = np.tile(np.repeat(b_grid, nE), nS * nF * nT)
    return beta, Pi_b


def bf_income(e_grid, thetaF, thetaI, a_grid, xi, ya):
    # Tested Income: y + ra, in units of E[y|F] = w * h_F
    wr = np.stack([np.exp(thetaF[:, None]) * e_grid,
                   xi * np.exp(thetaI[:, None]) * e_grid,             # xi = w_I h_I / (w h_F)
                   np.zeros((thetaF.size, e_grid.size))])             # Unemployed: always eligible
    wr = np.repeat(np.repeat(wr[:, None, :, None], nF, 1), nB, 3)     # s, b, theta, beta, e
    return wr.reshape(-1)[:, None] + ya * a_grid


def bf_test(yt, ybar, sig_BF, lambda_BF, rho_F, rho_I):
    # BF Test: P[b' = 1 | s, b, theta, beta, e, a]
    elig = expit((np.log(ybar) - np.log(yt + 1e-12)) / sig_BF)        # Steep 1{y < ybar}
    s, b = np.divmod(np.arange(yt.shape[0]) // (yt.shape[0] // (nS * nF)), nF)
    iu   = (s > 0)[:, None]
    # Entry: I and U only;     Audit: Formals always leave
    return np.where(b[:, None] == 0, lambda_BF * elig * iu,
                    1 - np.array([rho_F, rho_I, rho_I])[s][:, None] * (1 - elig * iu))


# 1.3. Labor Income Function
def labor_income(w, w_I, h_F, h_I, Tr, y_u, e_grid, nE, nT, thetaF, thetaI, tau_l, psi, varphi):
    # BF Status
    b = np.tile(np.repeat([0.0, 1.0], nT*nB*nE), nS)

    e_F = np.exp(thetaF[:, None]) * e_grid[None, :]
    e_I = np.exp(thetaI[:, None]) * e_grid[None, :]

    y_F = w   * e_F * h_F             # Gross Earnings
    y_I = w_I * e_I * h_I
    y_U = y_u * np.ones((nT, nE))     # Home Production

    y = np.r_[expand((1 - tau_l) * y_F),
              expand(y_I), expand(y_U)] + Tr * b

    vh = np.r_[expand(np.full((nT, nE), v(h_F, psi, varphi))),        # Effort Cost, by sector
               expand(np.full((nT, nE), v(h_I, psi, varphi))),
               expand(np.zeros((nT, nE)))]
    return y, vh, y_F, y_I, e_F, e_I, b


# ---------------------------------------------------------------------------
# 2. Endogenous Grid Method (EGM)
_HH_WARM = {}                     # cache in household_block.py
def hh_init(a_grid, y, r, eis):
    key = (y.shape[0], a_grid.shape[0])
    if key in _HH_WARM:           # reuse last converged guess to speed up
        return _HH_WARM[key]
    coh = (1 + r) * a_grid + np.maximum(y, 1e-12)[:, None]
    Va  = (1 + r) * uc(0.1 * coh, eis)
    V   = u(0.1 * coh, eis) / (1 - 0.96)
    Vc  = V.copy()
    return Va, V, Vc


@het(exogenous=['Pi'], policy='a', backward=['Va', 'V', 'Vc'], backward_init=hh_init)
def household(Va_p, V_p, Vc_p, a_grid, y, vh, r, beta, eis):
    # Change the SSJ Package to allow beta's heterogeneity
    c_nextgrid = (beta[:, None] * Va_p) ** (-eis)
    lhs = c_nextgrid + a_grid - y[:, None]
    c = interpolate.interpolate_y(lhs, (1 + r) * a_grid, c_nextgrid)

    a = np.clip((1 + r) * a_grid + y[:, None] - c, a_grid[0], a_grid[-1])
    c = (1 + r) * a_grid + y[:, None] - a         # at the bound: eat what is left

    Va = (1 + r) * uc(c, eis)
    V  = (u(c, eis) - vh[:, None] + beta[:, None] * interpolate.interpolate_y(a_grid, a, V_p))
    Vc = (u(c, eis) + beta[:, None] * interpolate.interpolate_y(a_grid, a, Vc_p))  # Without v(h), for CEV
    return Va, V, Vc, a, c


# ---------------------------------------------------------------------------
# 3. Hetoutputs
def sector_shares(c, e_F, e_I, h_F, h_I, b, beta, eis):
    sF, sI, sU = states(c)
    f, i, u, n_f, n_i, bf_f_lf, bf_i_lf, bf_u_lf, bf, uc_f, uc_i, beta_f, beta_i = zeros(c, 13)

    # Sector Indicator, F=0, I=1, U=2
    f[sF], i[sI], u[sU] = 1.0, 1.0, 1.0

    # Labor Supply = theta * e * h;  the Union sets the hours of both sectors
    n_f[sF] = expand(e_F * h_F)[:, None]
    n_i[sI] = expand(e_I * h_I)[:, None]

    # Bolsa Familia: Recipients
    bf = b[:, None] * np.ones_like(c)
    bf_f_lf[sF], bf_i_lf[sI], bf_u_lf[sU] = bf[sF], bf[sI], bf[sU]

    # Union: Efficiency-Weighted u'(c), Formals and Informals
    uc_f[sF]   = expand(e_F)[:, None] * uc(c, eis)[sF]
    uc_i[sI]   = expand(e_I)[:, None] * uc(c, eis)[sI]
    beta_f[sF] = beta[:, None][sF] * uc_f[sF]
    beta_i[sI] = beta[:, None][sI] * uc_i[sI]

    return f, i, u, n_f, n_i, bf_f_lf, bf_i_lf, bf_u_lf, bf, uc_f, uc_i, beta_f, beta_i


def labor_moments(a, c, y_F, y_I, y, r, a_grid):
    # Moments for Calibration.
    sF, sI, _ = states(c)
    log_y_f, log_y_i = zeros(c, 2)

    log_y_f[sF] = expand(np.log(y_F))[:, None]
    log_y_i[sI] = expand(np.log(y_I))[:, None]

    # MPC out of a transitory transfer: 1 - da'/dcoh, with dcoh = (1+r) da.
    mpc = 1 - np.diff(a, axis=1) / ((1 + r) * np.diff(a_grid))
    mpc = np.c_[mpc, mpc[:, -1]]

    inc_l = y[:, None] * np.ones_like(c)      # all households, non-financial income
    mpc_y = mpc * inc_l

    return log_y_f, log_y_i, mpc_y, inc_l


# ---------------------------------------------------------------------------
# 5. Household Block
hh    = household.add_hetinputs([make_egrid, make_bgrid, labor_income])
hh_ss = hh.add_hetoutputs([sector_shares, labor_moments])
hh    = hh.add_hetoutputs([sector_shares])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')
