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
from sequence_jacobian import het, interpolate, grids

random.seed(20260415)


# ---------------------------------------------------------------------------
# 1. Utility, Grid, and Income

# 1.1. Utility Functions
nB, nS = 2, 3           # beta_grid x labor_grid size
u = lambda c, eis: np.log(np.maximum(c, 1e-12)) if eis == 1 else\
                            np.maximum(c, 1e-12)  ** (1-1/eis) / (1-1/eis)
v = lambda h, psi, varphi: psi * np.maximum(h, 1e-8) ** (1+1/varphi)/(1+1/varphi)

expand = lambda x: np.repeat(x[:, None, :], nB, 1).reshape(-1)   # (nT, nE) -> state


# 1.2. Exogenous Transition States Grid
def discretize_normal(mu, sigma, n):
    # theta_s ~ N(mu_s, sigma_s^2) as Gauss-Hermite quadrature with nT nodes.
    z, w = np.polynomial.hermite.hermgauss(n)
    theta = mu + np.sqrt(2) * sigma * z
    prob  = w / np.sqrt(np.pi)
    return theta, prob


def make_egrid(rho_e, sd_e, nE, amin, amax, nA,
               sigma_F, mu_I, sigma_I, nT):
    # Productivity Grids.
    e_grid, pi_e_e, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    e_grid = e_grid / np.sum(pi_e_e * e_grid)
    thetaF, probF = discretize_normal(0.0, sigma_F, nT)
    thetaF = thetaF - np.log(probF @ np.exp(thetaF))   # Normalized: E(exp(theta_F)) = 1
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
    beta = np.tile(np.repeat(b_grid, nE), nS * nT)
    return beta, Pi_b


# 1.3. Labor Income Function
def labor_income(w, h_F, Div, Tr, tau, e_grid, nE, nT, thetaF,
                 thetaI, y_bar, tau_l, psi, varphi, sig_y):
    # Dividend Income and Informal Hours
    div_i = np.tile(Div * e_grid, nS*nT*nB)
    tau_i = np.tile(tau, nS*nT*nB*nE)

    e_F = np.exp(thetaF[:, None]) * e_grid[None, :]
    e_I = np.exp(thetaI[:, None]) * e_grid[None, :]

    h_I = (w * e_I / np.maximum(psi, 1e-8)) ** (varphi)
    y_F = w * e_F * h_F                       # Gross Earnings
    y_I = w * e_I * h_I
    elig = (y_I < y_bar).astype(float)        # Elegibility
    # elig = 1 / (1 + np.exp((y_I - y_bar) / sig_y))

    # Expand the income into beta grid.
    y = np.r_[expand((1 - tau_l) * y_F),
              expand(1/(1+varphi) * y_I + Tr * elig),
              expand(np.full((nT, nE), Tr))] + div_i + tau_i
    return y, y_F, y_I, h_I, e_F, e_I, elig


# ---------------------------------------------------------------------------
# 2. Endogenous Grid Method (EGM)
_HH_WARM = {}                     # cache in household_block.py
def hh_init(a_grid, y, r, eis):
    key = (y.shape[0], a_grid.shape[0])
    if key in _HH_WARM:           # reuse last converged guess to speed up
        return _HH_WARM[key]
    coh = (1 + r) * a_grid + y[:, None]
    Va  = (1 + r) * (0.1 * coh) ** (-1 / eis)
    V   = u(0.1 * coh, eis) / (1 - 0.96)
    return Va, V


@het(exogenous=['Pi'], policy='a', backward=['Va', 'V'], backward_init=hh_init)
def household(Va_p, V_p, a_grid, y, h_F, r, beta, eis, psi, varphi):
    c_nextgrid = (beta[:, None] * Va_p) ** (-eis)
    coh = (1 + r) * a_grid + y[:, None]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])
    c_ghh = coh - a
    Va = (1 + r) * c_ghh ** (-1 / eis)
    # Value Function
    dis = np.zeros_like(c_ghh)
    dis[:c_ghh.shape[0] // 3] = v(h_F, psi, varphi)
    V = u(c_ghh, eis) - dis + beta[:, None] * V_p
    return Va, V, a, c_ghh


# ---------------------------------------------------------------------------
# 3. Hetoutputs
def sector_shares(c_ghh, e_F, e_I, h_F, h_I, elig, psi, varphi):
    block = c_ghh.shape[0] // nS     # nT * nBeta * nE
    f, n_f = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    i, n_i = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    u, bf  = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    
    # Sector Indicator, F=0, I=1, U=2
    f[:block]        = 1.0
    i[block:2*block] = 1.0
    u[2*block:]      = 1.0
    bf[2*block:]     = 1.0

    # Labor Supply = theta * e * h
    n_f[:block]        = expand(e_F * h_F)[:, None]
    n_i[block:2*block] = expand(e_I * h_I)[:, None]
    bf[block:2*block]  = expand(elig)[:, None]

    # Informal Consumption (readjusted)
    c = c_ghh.copy()
    c[block:2*block] += expand(v(h_I, psi, varphi))[:, None]

    return c, f, i, u, n_f, n_i, bf


def labor_moments(c_ghh, y_F, y_I, h_I):
    # Moments for Calibration.
    block = c_ghh.shape[0] // nS
    h_i = np.zeros_like(c_ghh)          # Informal Hours
    log_y_f, log_y_i = (np.zeros_like(c_ghh) for _ in range(2))

    h_i[block:2*block]     = expand(h_I)[:, None]
    log_y_f[:block]        = expand(np.log(y_F))[:, None]
    log_y_i[block:2*block] = expand(np.log(y_I))[:, None]

    return h_i, log_y_f, log_y_i


# ---------------------------------------------------------------------------
# 5. Household Block

hh = household.add_hetinputs([make_egrid, make_bgrid, labor_income])
hh = hh.add_hetoutputs([sector_shares, labor_moments])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')
