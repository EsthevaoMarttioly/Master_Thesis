#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid, and transition matrices for income, labor productivity, and sectoral status.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import numpy as np
import random
from scipy.stats import norm
from scipy.special import expit
from sequence_jacobian import het, interpolate, grids

random.seed(20260415)


# ---------------------------------------------------------------------------
# 1. Utility, Grid, and Income

# 1.1. Utility Functions
nB, nS, nF = 2, 3, 2    # beta_grid x labor_grid x BF status size

expand = lambda x: np.tile(np.repeat(x[:, None, :], nB, 1).reshape(-1), nF)
zeros  = lambda x, n: [np.zeros_like(x) for _ in range(n)]
states = lambda x: [slice(k * x.shape[0] // nS, (k+1) * x.shape[0] // nS) for k in range(nS)]

u = lambda c, eis: np.log(np.maximum(c, 1e-12)) if eis == 1 else\
                            np.maximum(c, 1e-12)  ** (1-1/eis) / (1-1/eis)

v = lambda h, psi, varphi: psi * np.maximum(h, 1e-8) ** (1+1/varphi)/(1+1/varphi)

def cn(uc, we, eis, varphi, psi, h_F):      # Euler + Intertemporal FOC (capped)
    return uc ** (-eis), np.minimum((we * uc / psi) ** varphi, h_F)

def solve_cn(we, res, eis, varphi, psi, h_F, uc):
    # If constrained (a' = amin), c - we*h(c) = res.
    # If not, Newton in log u'(c), whose derivative [-eis*c - we*varphi*h] is always negative.
    x = np.log(uc)
    for _ in range(30):
        c, h = cn(np.exp(x), we, eis, varphi, psi, h_F)
        ne   = c - we * h - res
        if np.max(np.abs(ne)) < 1e-11: break
        x   -= ne / (-eis * c - we * varphi * h * (h < h_F))
    return cn(np.exp(x), we, eis, varphi, psi, h_F)


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


def bf_income(e_grid, thetaF, thetaI, a_grid, ya):
    # Tested Income: y + ra, in units of E[y|F] = w * h_F
    wr = np.stack([np.exp(thetaF[:, None]) * e_grid,
                   np.exp(thetaI[:, None]) * e_grid,
                   np.zeros((thetaF.size, e_grid.size))])            # Unemployed: no wage
    wr = np.repeat(np.repeat(wr[:, None, :, None], nF, 1), nB, 3)    # s, b, theta, beta, e
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
def labor_income(w, h_F, Tr, tau, e_grid, nE, nT, thetaF, thetaI, tau_l, psi, varphi):
    # Lump-Sum Rebate and BF Status
    tau_i = np.tile(tau, nS*nF*nT*nB*nE)
    b     = np.tile(np.repeat([0.0, 1.0], nT*nB*nE), nS)

    e_F = np.exp(thetaF[:, None]) * e_grid[None, :]
    e_I = np.exp(thetaI[:, None]) * e_grid[None, :]

    y_F = w * e_F * h_F           # Gross Earnings

    # Income that doesn't come from own hours
    y = np.r_[expand((1 - tau_l) * y_F),
              expand(np.zeros((nT, nE))),
              expand(np.zeros((nT, nE)))] + Tr * b + tau_i

    we = np.r_[expand(np.zeros((nT, nE))),        # Zero because it's not chosen
               expand(w * e_I),                   # Marginal Return to Hours
               expand(np.zeros((nT, nE)))]
    
    vh = np.r_[expand(np.full((nT, nE), v(h_F, psi, varphi))),      # Formal Effort Cost
               expand(np.zeros((nT, nE))),
               expand(np.zeros((nT, nE)))]
    return y, we, vh, y_F, e_F, e_I, b


# ---------------------------------------------------------------------------
# 2. Endogenous Grid Method (EGM)
_HH_WARM = {}                     # cache in household_block.py
def hh_init(a_grid, y, we, r, eis):
    key = (y.shape[0], a_grid.shape[0])
    if key in _HH_WARM:           # reuse last converged guess to speed up
        return _HH_WARM[key]
    coh = (1 + r) * a_grid + (y + we)[:, None]
    Va  = (1 + r) * (0.1 * coh) ** (-1 / eis)
    V   = u(0.1 * coh, eis) / (1 - 0.96)
    return Va, V


@het(exogenous=['Pi'], policy='a', backward=['Va', 'V'], backward_init=hh_init)
def household(Va_p, V_p, a_grid, y, we, vh, r, beta, eis, varphi, psi, h_F):
    # Change the SSJ Package to allow beta's heterogeneity
    uc_nextgrid = beta[:, None] * Va_p
    c_nextgrid, h_nextgrid = cn(uc_nextgrid, we[:, None], eis, varphi, psi, h_F)

    # Endogenous grid: resources the choice needs, against the ones carried in
    lhs = c_nextgrid - we[:, None] * h_nextgrid + a_grid - y[:, None]
    c = interpolate.interpolate_y(lhs, (1 + r) * a_grid, c_nextgrid)
    h = interpolate.interpolate_y(lhs, (1 + r) * a_grid, h_nextgrid)

    a = (1 + r) * a_grid + we[:, None] * h + y[:, None] - c
    ab = np.clip(a, a_grid[0], a_grid[-1])
    bd = ab != a                      # a bound binds: redo the static problem at a'
    if bd.any():
        res = (1 + r) * a_grid + y[:, None] - ab
        c[bd], h[bd] = solve_cn(np.broadcast_to(we[:, None], a.shape)[bd], res[bd],
                                eis, varphi, psi, h_F, uc_nextgrid[bd])
    a = ab

    Va = (1 + r) * c ** (-1 / eis)
    V  = (u(c, eis) - v(h, psi, varphi) - vh[:, None]
          + beta[:, None] * interpolate.interpolate_y(a_grid, a, V_p))
    return Va, V, a, c, h


# ---------------------------------------------------------------------------
# 3. Hetoutputs
def sector_shares(c, h, e_F, e_I, h_F, b, beta, eis):
    sF, sI, sU = states(c)
    f, i, u, n_f, n_i, bf_f_lf, bf_i_lf, bf_u_lf, bf, uc_f, beta_f = zeros(c, 11)

    # Sector Indicator, F=0, I=1, U=2
    f[sF], i[sI], u[sU] = 1.0, 1.0, 1.0

    # Labor Supply = theta * e * h;  Formal Hours set by the Union
    n_f[sF] = expand(e_F * h_F)[:, None]
    n_i[sI] = expand(e_I)[:, None] * h[sI]

    # Bolsa Familia: Recipients
    bf = b[:, None] * np.ones_like(c)
    bf_f_lf[sF], bf_i_lf[sI], bf_u_lf[sU] = bf[sF], bf[sI], bf[sU]

    # Union: Efficiency-Weighted u'(c) over Formals
    uc_f[sF]   = expand(e_F)[:, None] * c[sF] ** (-1/eis)
    beta_f[sF] = beta[:, None][sF] * uc_f[sF]

    return f, i, u, n_f, n_i, bf_f_lf, bf_i_lf, bf_u_lf, bf, uc_f, beta_f


def labor_moments(a, c, h, y_F, we, r, a_grid, tau_l):
    # Moments for Calibration.
    sF, sI, _ = states(c)
    h_i, log_y_f, log_y_i = zeros(c, 3)

    h_i[sI]     = h[sI]
    log_y_f[sF] = expand(np.log(y_F))[:, None]
    log_y_i[sI] = np.log(np.maximum(we[:, None] * h, 1e-12))[sI]

    # MPC out of a transitory transfer: 1 - da'/dcoh, with dcoh = (1+r) da.
    sF, _, _ = states(a)
    mpc = 1 - np.diff(a, axis=1) / ((1 + r) * np.diff(a_grid))
    mpc = np.c_[mpc, mpc[:, -1]]

    inc_l = we[:, None] * h                            # Informal: hours are chosen
    inc_l[sF] = expand((1 - tau_l) * y_F)[:, None]     # Formal:   hours set by the union
    mpc_y = mpc * inc_l

    return h_i, log_y_f, log_y_i, mpc_y, inc_l


# ---------------------------------------------------------------------------
# 5. Household Block
hh    = household.add_hetinputs([make_egrid, make_bgrid, labor_income])
hh_ss = hh.add_hetoutputs([sector_shares, labor_moments])
hh    = hh.add_hetoutputs([sector_shares])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')
