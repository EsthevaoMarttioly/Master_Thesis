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

from sequence_jacobian import het, hetblocks, interpolate, grids


# Set a seed for future replications
random.seed(20260415)


# ---------------------------------------------------------------------------
# 1. Endogenous Grid Method (EGM) for the HH problem
@het(exogenous=['Pi'], policy='a', backward='Va', backward_init=hetblocks.hh_sim.hh_init)
def household(Va_p, a_grid, y, r, beta, eis):
    """Slightly modify hetblocks.hh_sim.hh to allow for beta vector"""
    c_nextgrid = (beta[:, np.newaxis] * Va_p) ** (-eis)
    coh        = (1 + r) * a_grid + y[..., np.newaxis]

    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])     # borrowing constraint
    c = coh - a
    Va = (1 + r) * c ** (-1/eis)

    return Va, a, c


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

    # Check if rows sum to 1
    assert np.allclose(Pi_s.sum(axis=1), 1.0), "Pi_s rows must sum to 1"
    assert np.all(Pi_s >= 0), "Pi_s has negative entries"


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


# 2.2. Dividend Income Function
def dividend_income(Div, e_grid, nS):
    # Distribute firm profits proportional to productivity e
    div_e   = Div * e_grid              # sums to ~Div
    div_inc = np.tile(div_e, nS*2)      # tile across s*beta blocks
    return div_inc


# 2.3. Optimal Informal Hours
def informal_hours(e_grid, w_I, psi, varphi):
    # Intratemporal FOC:  psi * h^(1/varphi) = w_I * e   (no wealth effect on h)
    h_I = (w_I * e_grid / psi) ** (varphi)
    return h_I


# 2.4. Labor Income Function
def labor_income(e_grid, w, w_I, h_I, y_bar, tau_l, Tr, div_inc, varphi):
    Tr_inform = (w_I * e_grid * h_I < y_bar).astype(float)

    y_form   = (1 - tau_l) * w * e_grid                             # [formal]
    y_inform = 1/(1+varphi) * w_I * h_I * e_grid + Tr * Tr_inform   # [informal]
    y_unemp  = Tr * np.ones_like(e_grid)                            # [unemployed]

    y = np.r_[np.tile(y_form, 2),      # s=0, F
              np.tile(y_inform, 2),    # s=1, I
              np.tile(y_unemp, 2),     # s=2, U
              ] + div_inc              # asset income
    return y, Tr_inform


# ---------------------------------------------------------------------------
# 3. Hetoutputs

# Employment Status: 0=formal, 1=informal, 2=unemployed
def formal(c):
    nS = c.shape[0] // 3      # nBeta * nE per s-block
    f  = np.zeros_like(c)
    f[:nS, :] = 1.0         # s=0 block
    return f


def informal(c, e_grid, h_I):
    nS = c.shape[0] // 3
    i  = np.zeros_like(c)
    i[nS : 2*nS, :] = 1.0     # s=1 block

    # Informal Labor Supply: h_I * e for s=I
    n_i = np.zeros_like(c)
    n_i[nS : 2*nS, :] = np.tile(e_grid * h_I, 2)[:, np.newaxis]
    return i, n_i


def unemp(c):
    nS = c.shape[0] // 3
    u  = np.zeros_like(c)
    u[2*nS:, :] = 1.0         # s=2 block
    return u


def bolsa_familia(c, Tr_inform):
    nS = c.shape[0] // 3
    bf = np.zeros_like(c)
    bf[nS : 2*nS, :] = np.tile(Tr_inform, 2)[:, np.newaxis]
    bf[2*nS:, :] = 1.0
    return bf



# ---------------------------------------------------------------------------
# 4. Household Block

hh = household.add_hetinputs([make_grid, dividend_income, informal_hours, labor_income])
hh = hh.add_hetoutputs([formal, informal, unemp, bolsa_familia])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')


# from code.parameters import calibration

# hh.steady_state(calibration)
