#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid and transition matrices for income, and the labor income function.
# ---------------------------------------------------------------------------
#=

# Import Packages
import random
import numpy as np

from sequence_jacobian import het, interpolate, grids


# Set a seed for future replications
random.seed(20260415)


# ---------------------------------------------------------------------------
# 1. EGM Problem

## Marginal Value of Assets:           V_a = (1+r) * c^{-1/eis}
## Cash on Hand = (1+r)*a + y(s,e);    Arbitrary guess: c = y + r*a
def household_init(a_grid, y, r, eis):
    c = np.maximum(1e-8, y[..., np.newaxis] + np.maximum(r, 0.04) * a_grid)
    Va = (1 + r) * (c ** (-1 / eis))
    return Va


## Endogenous Grid Method (EGM) for the HH problem
@het(exogenous=['Pi'], policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, y, r, beta, eis):
    """Single backward iteration step using EGM.

    Euler equation: u'(c_t) = beta * E[Va_{t+1}]
    Cash-on-hand:   CoH = (1+r)*a + y(s,beta,e)

    Va : (nState, nA)  marginal value of assets today
    a  : (nState, nA)  asset policy
    c  : (nState, nA)  consumption policy"""

    c_nextgrid = (beta[:, np.newaxis] * Va_p) ** (-eis)
    coh = (1 + r) * a_grid + y[..., np.newaxis]

    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])     # borrowing constraint
    c = coh - a
    Va = (1 + r) * c ** (-1/eis)

    return Va, a, c


# ---------------------------------------------------------------------------
# 2. Hetinputs

# 2.1. Transition States Grid
def make_grid(rho_e, sd_e, nE, amin, amax, nA,
              beta_high, dbeta, omega_I, q, f, s, Tb):
    """Build all grids and the joint Markov transition matrix Pi.

    e_grid  : (nE,)            productivity grid
    pi_e_e  : (nE,)            stationary distribution of e (Rouwenhorst).
    Pi      : (nS*nBeta*nE)^2  full Kronecker transition matrix
    a_grid  : (nA,)            log-spaced asset grid
    beta    : (nS*nBeta*nE,)   per-state discount factor"""

    e_grid, pi_e_e, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)   # Log-spaced grid for assets

    # ------------------------------------------------------------------
    # Employment Transition: 0=employed, 1=unemployed, 2=needy
    Pi_s = np.vstack(([1 - s,   s,              0.0 ],   # Pi_s[E,U] = s    (loses job)
                      [f,       1 - f - 1/Tb,   1/Tb],   # Pi_s[U,V] = 1/Tb (loses benefit)
                      [f,       0.0,            1-f ]))  # Pi_s[V,E] = f    (finds job)

    # Check if rows sum to 1
    assert np.allclose(Pi_s.sum(axis=1), 1.0), "Pi_s rows must sum to 1"
    assert np.all(Pi_s >= 0), \
        f"Pi_s has negative entries - check that Tb >= {1/(1-f):.2f} for f={f}"


    # ------------------------------------------------------------------
    # Discount Factor Transition  (impatient=0, patient=1)
    beta_low = beta_high - dbeta
    b_grid   = np.array([beta_low, beta_high])
    pi_b     = np.array([omega_I, 1-omega_I])    # stationary shares
    Pi_b     = (1 - q) * np.eye(2) + q * np.outer(np.ones(2), pi_b)


    # ------------------------------------------------------------------
    # Kronecker:  s (3)  \otimes  beta (2)  \otimes  e (nE)
    Pi_be = np.kron(Pi_b, Pi_e)      # (beta, e)
    Pi    = np.kron(Pi_s, Pi_be)     # (s, beta, e)

    # beta vector: repeat [beta_low]*nE, [beta_high]*nE for each of 3 s-blocks
    beta = np.tile(np.repeat(b_grid, nE), 3)   # (s, beta, e)

    return e_grid, pi_e_e, Pi, a_grid, beta


# 2.2. Dividend Income Function
def dividend_income(e_grid, pi_e_e, Div, tau):
    """Distribute firm profits proportional to productivity e"""
    E_e     = np.sum(pi_e_e * e_grid)          # E[e] under Rouwenhorst stationary dist.
    div_e   = Div * e_grid / E_e               # (nE,): per-e share, sums to ~Div
    div_inc = np.tile(div_e, 6)                # tile across 3 s-blocks * 2 beta-types
    return div_inc


# 2.3. Labor Income Function
def labor_income(e_grid, w, b, tau, Tr, div_inc):
    # Set grid length
    nE_ones = np.ones(len(e_grid))

    y_emp   = (1 - tau) * w * e_grid    # [employed]
    y_unemp = b * nE_ones               # [unemployed]
    y_needy = Tr * nE_ones              # [needy]

    y = np.r_[np.tile(y_emp, 2),     # s=0, E
              np.tile(y_unemp, 2),   # s=1, U (with benefits)
              np.tile(y_needy, 2)    # s=2, V (needy)
              ] + div_inc            # equity endowment
    return y


# ---------------------------------------------------------------------------
# 3. Hetoutputs

# Employment Status: 0=employed, 1=unemployed, 2=needy
def unemployment(c):
    N  = c.shape[0]           # nS * nBeta * nE
    nS = N // 3               # nBeta * nE per s-block
    u  = np.zeros_like(c)
    u[nS : 2 * nS, :] = 1.0   # s=1 block
    return u


def needy(c):
    N  = c.shape[0]
    nS = N // 3
    v  = np.zeros_like(c)
    v[2 * nS :, :] = 1.0      # s=2 block
    return v


# ---------------------------------------------------------------------------
# 4. Household Block

hh = household.add_hetinputs([make_grid, dividend_income, labor_income])
hh = hh.add_hetoutputs([unemployment, needy])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')


# from code.parameters import calibration, unknowns_ss

# calibration["w"] = 0.9
# calibration["Tr"] = 0.1
# calibration["beta_high"] = 0.05
# calibration["Tb"] = 3
# calibration["b"] = 0.5
# calibration["Div"] = calibration["Y"] - calibration["w"] * calibration["Z"] / calibration["mu"]

# hh.steady_state(calibration)
