#=
#---------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid and transition matrices for income, and the labor income function.
#---------------------------------------------------------------------------
#=

# Import Packages
import random
import numpy as np

from sequence_jacobian import het, interpolate, grids


# Set a seed for future replications
random.seed(20260415)


## 1. EGM Problem
## Initial guess for marginal value of assets: V_a = (1+r) * c^{-1/eis}
## Cash on Hand = (1+r)*a + y(s,e);    Arbitrary guess: c = y + r*a
def household_init(a_grid, y, r, eis):
    c = np.maximum(1e-8, y[..., np.newaxis] + np.maximum(r, 0.04) * a_grid)
    Va = (1 + r) * (c ** (-1 / eis))
    return Va


## Endogenous Grid Method (EGM) for the HH problem
@het(exogenous=['Pi'], policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, y, r, beta, eis):
    """Single backward iteration step using EGM.
    Va_p     : array (nE, nA), expected marginal value of assets next period
    Va       : array (nE, nA), marginal value of assets today
    a_grid   : array (nA), asset grid
    a        : array (nE, nA), asset policy today
    c        : array (nE, nA), consumption policy today"""

    c_nextgrid = (beta[:, np.newaxis] * Va_p) ** (-eis)  # u'(ct+1) = beta * E(Va^(t+1)) = c_{t+1}^(-1/eis)
    coh = (1 + r) * a_grid + y[..., np.newaxis]

    # We solve a as function of CoH, but interpolating on the grid
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])          # a >= amin
    c = coh - a                           # c + a' = (1+r)*a + y = CoH
    Va = (1 + r) * c ** (-1/eis)          # Va^t = (1+r) * u'(c_t)

    return Va, a, c



## 2. Grid, Transition Matrices and Income
def make_grid(rho_e, sd_e, nE, amin, amax, nA,
              beta_high, dbeta, lambda_I, q, f, s, Tb):
    # Grid for idiosyncratic productivity and assets
    e_grid, _, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)   # Log-spaced grid for assets

    # Employment status: 0=employed, 1=unemployed, 2=needy
    Pi_s = np.vstack(([1 - s,   s,              0   ],   # Pi_s[E,U] = s    (loses job)
                      [f,       1 - f - 1/Tb,   1/Tb],   # Pi_s[U,V] = 1/Tb (loses benefit)
                      [f,       0,              1-f ]))  # Pi_s[V,E] = f    (finds job)
    
    # Check if rows sum to 1
    assert np.allclose(Pi_s.sum(axis=1), 1.0), "Pi_s is not row-stochastic"
 

    # Beta grid: Impatient (beta_low) and Patient (beta_high)
    beta_low = beta_high - dbeta
    b_grid   = np.array([beta_low, beta_high])
    pi_b     = np.array([lambda_I, 1 - lambda_I])           # stationary shares
    Pi_b     = (1 - q) * np.eye(2) + q * np.outer(np.ones(2), pi_b)


    # Kronecker: outer = s (3), middle = beta (2), inner = e (nE)
    Pi_be = np.kron(Pi_b, Pi_e)      # (beta, e)
    Pi    = np.kron(Pi_s, Pi_be)     # (s, beta, e)
 
    # For each s-block: [b0]*nE, [b1]*nE  repeated for both s values
    beta = np.tile(np.repeat(b_grid, nE), 3)   # (s, beta, e)

    return e_grid, Pi, a_grid, beta



def labor_income(e_grid, w, b, tau, Tr):
    # Set grid length
    nE_ones = np.ones(len(e_grid))

    y_emp   = (1 - tau) * w * e_grid + Tr * nE_ones  # [employed]
    y_unemp = b * nE_ones                            # [unemployed]
    y_needy = Tr * nE_ones                           # [needy]

    y = np.r_[np.tile(y_emp, 2),     # s=0, E
              np.tile(y_unemp, 2),   # s=1, U (with benefits)
              np.tile(y_needy, 2)]   # s=2, V (needy)
    return y



## 3. The employment status: 0=employed, 1=unemployed, 2=needy
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



## 4. The Household Block
hh = household.add_hetinputs([make_grid, labor_income])
hh = hh.add_hetoutputs([unemployment, needy])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')


# from code.parameters import calibration, unknowns_ss

# calibration["w"] = 0.9
# calibration["Tr"] = 0.1
# calibration["beta_high"] = 0.05
# calibration["Tb"] = 3
# calibration["b"] = 0.5

# hh.steady_state(calibration)
