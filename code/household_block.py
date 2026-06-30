#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid, and transition matrices for income, labor productivity, and sectoral status.
# ---------------------------------------------------------------------------
#=

# Import Packages
import random
import numpy as np

from sequence_jacobian import het, interpolate, grids


# Set a seed for future replications
random.seed(20260415)


# ---------------------------------------------------------------------------
# 1. Endogenous Grid Method (EGM) for the HH problem
def hh_init(a_grid, y, r, eis):
    coh = (1.0 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va  = (1.0 + r) * (0.1 * coh) ** (-1/eis)
    V   = (0.1 * coh) ** (1-1/eis) / (1-1/eis) / (1 - 0.96)
    return Va, V


@het(exogenous=['Pi'], policy='a', backward=['Va','V'], backward_init=hh_init)
def household(Va_p, V_p, a_grid, y, r, beta, eis):
    """Slightly modify hetblocks.hh_sim.hh to allow for beta vector"""
    c_nextgrid = (beta[:, np.newaxis] * Va_p) ** (-eis)
    coh        = (1.0 + r) * a_grid + y[..., np.newaxis]

    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])     # borrowing constraint
    c_ghh = coh - a
    Va = (1.0 + r) * c_ghh ** (-1/eis)
    V = c_ghh**(1-1/eis)/(1-1/eis) + beta[:, np.newaxis] * V_p

    return Va, V, a, c_ghh


# ---------------------------------------------------------------------------
# 2. Hetinputs

# 2.1. Transition States Grid
def make_grid(rho_e, sd_e, nE, amin, amax, nA,
              beta_high, dbeta, omega_I, q,
              p_fi, p_if, p_iu, p_uf, p_ui):
    """Build all grids and the joint Markov transition matrix Pi.

    e_grid  : (nE,)            productivity grid, normalized E[e] = 1
    Pi      : (nS*nBeta*nE)^2  full Kronecker transition matrix
    a_grid  : (nA,)            log-spaced asset grid
    beta    : (nS*nBeta*nE,)   per-state discount factor"""

    e_grid, pi_e_e, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    e_grid = e_grid / np.sum(pi_e_e * e_grid)
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)

    # ------------------------------------------------------------------
    # Employment Transition: 0=formal, 1=informal, 2=unemployed
    Pi_s = np.vstack(([1 - p_fi,  p_fi,             0               ],   # Pi_s[F,I] = p_fi  (formal to informal)
                      [p_if,      1 - p_if - p_iu,  p_iu            ],   # Pi_s[I,F] = p_if  (informal to formal)
                      [p_uf,      p_ui,             1 - p_uf - p_ui ]))  # Pi_s[U,F] = p_uf  (finds formal job)


    # ------------------------------------------------------------------
    # Discount Factor Transition  (impatient=0, patient=1)
    beta_low = beta_high - dbeta
    b_grid   = np.array([beta_low, beta_high])
    pi_b     = np.array([omega_I, 1-omega_I])    # stationary shares
    Pi_b     = (1 - q) * np.eye(2) + q * np.outer(np.ones(2), pi_b)


    # ------------------------------------------------------------------
    # Kronecker:  s (3)  \otimes  beta (2)  \otimes  e (nE)
    nS    = len(Pi_s)
    Pi_be = np.kron(Pi_b, Pi_e)      # (beta, e)
    Pi    = np.kron(Pi_s, Pi_be)     # (s, beta, e)

    # beta vector: repeat [beta_low]*nE, [beta_high]*nE for each of s-blocks
    beta = np.tile(np.repeat(b_grid, nE), nS)   # (s, beta, e)

    return e_grid, a_grid, beta, Pi, nS


# 2.2. Optimal Informal Hours
def informal_hours(e_grid, w_I, psi, varphi):
    # Intratemporal FOC, No Wealth Effect
    h_I = (w_I * e_grid / psi) ** (varphi)
    return h_I


# 2.3. Labor Income Function
def labor_income(w, w_I, h_F, h_I, Div, Tr, e_grid, nS, y_bar, tau_l, varphi):
    div_e = np.tile(Div * e_grid, nS*2)     # Asset Income

    # BF Elegibility
    Tr_inform = (w_I * e_grid * h_I < y_bar).astype(float)

    # Informal: w_I * e_grid * h_I - v(h) = 1/(1+varphi) * w_I * e_grid * h_I
    y_formal = np.tile((1 - tau_l) * w * e_grid * h_F, 2)                        # [formal]
    y_inform = np.tile(1/(1+varphi) * w_I * h_I * e_grid + Tr * Tr_inform, 2)    # [informal]
    y_unemp  = np.tile(Tr * np.ones_like(e_grid), 2)                             # [unemployed]

    y = np.r_[y_formal, y_inform, y_unemp] + div_e

    return y, Tr_inform


# ---------------------------------------------------------------------------
# 3. Hetoutputs
def sector_shares(c_ghh, e_grid, h_F, h_I, Tr_inform):
    nS = c_ghh.shape[0] // 3    # nBeta * nE per s-block

    # Formal: s=0;      Formal Supply = e * h_F
    f, n_f = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    i, n_i = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    u, bf  = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    
    # Sector Indicator, F=0, I=1, U=2
    f[:nS, :]       = 1.0
    i[nS : 2*nS, :] = 1.0
    u[2*nS:, :]     = 1.0
    bf[2*nS:, :]    = 1.0

    # Labor Supply = e * h
    n_f[:nS, :] = np.tile(e_grid * h_F, 2)[:, np.newaxis]
    n_i[nS : 2*nS, :] = np.tile(e_grid * h_I, 2)[:, np.newaxis]
    bf[nS : 2*nS, :] = np.tile(Tr_inform, 2)[:, np.newaxis]

    return f, i, u, n_f, n_i, bf



# ---------------------------------------------------------------------------
# 4. Household Block

hh = household.add_hetinputs([make_grid, informal_hours, labor_income])
hh = hh.add_hetoutputs([sector_shares])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')


# from code.parameters import *

# calibration_ss = calibration_ss | dict(psi = 1.1, beta_high = 0.96)

# hh.steady_state(calibration_ss).internals['household'].keys()


