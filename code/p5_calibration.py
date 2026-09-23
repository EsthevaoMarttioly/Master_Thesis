#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Define the external and internal parameters and the SMM estimator.
# ---------------------------------------------------------------------------
#=

import os
import numpy as np
import pandas as pd

USE_SMM = False     # True: Use SMM estimates if available, False: Use hand-set guesses


# ---- Data and Helpers -----------------------------------------------------
Pi_s          = pd.read_csv('data/final/pnad_transition_matrix.csv', index_col=0).iloc[:3].values
pnad, pnad_se = [r for _, r in pd.read_csv('data/final/pnad_calibration.csv').iterrows()]
alpha         = np.array(pnad[['F', 'I', 'U']])
qs = (10, 25, 50, 75, 90)      # Income Quantiles
F, I, U = 0, 1, 2


# Arrival Rates and Flows Calibration
pi_calib = {'pi_F': (I, F), 'pi_I': (F, I), 'pi_UF': (U, F), 'pi_UI': (U, I)}
sectors  = {'FI': (F, I), 'FU': (F, U), 'IF': (I, F),
            'IU': (I, U), 'UF': (U, F), 'UI': (U, I)}
flows    = {n: Pi_s[i, j] for n, (i, j) in sectors.items()}


# BF Calibration and Initial Guesses
bf_calib = {'lambda_BF': ('BF_U_LF', 'U', 1),      # (moment, sector, sign)
            'ybar'     : ('BF_I_LF', 'I', 1),
            'rho_F'    : ('BF_F_LF', 'F', -1)}
unknowns = dict(beta_high = 0.98, psi = 0.7, L = 0.7, tau = 0.1, Tr = 0.2, B = 4.0)
update   = [*pi_calib, *bf_calib, *unknowns, 'tau_ss', 'B_ss']


# External Moments for SMM
wid = dict(gini  = 0.82,             # 2025 Global Wealth Report, UBS
           bot50 = 0.02,             # Brazilian Wealth Shares (WID.world, 2024), untargeted
           top10 = 0.719,
           top1  = 0.395,
           htm   = 0.35,             # Costa-Junior et al. (2025), untargeted
           mpc   = 0.20,             # Income-Weighted Quarterly MPC: Auclert et al. (2025)
           A_gdp = 14.82 / 12.7,     # M4 / GDP (BCB, IBGE)
        #    A_gdp = 7.90 / 12.7,      # Household Financial Investments / GDP (ANBIMA, IBGE)
)


# ---------------------------------------------------------------------------
# 1. External Calibration
calibration = dict(
    # --- Household Preferences ---
    eis    = 0.5,                 # EIS = gamma = 0.5 (CRRA sigma = 2)
    varphi = 0.5,                 # Frisch Elasticity             --- attention!
    h_F    = 1.0,                 # Normalized: Formal Worked Hours

    # --- Discount Factor ---
    dbeta     = 0.16,    # SMM: beta_high - beta_low           -> Wealth
    omega_I   = 0.70,    # SMM: Share of Impatient Agents      -> HtM
    q         = 0.01,    # Prob of Redrawing beta Type (Generation = 25y = 100q)

    # --- Labor market ---
    delta_F = Pi_s[F, U],   # Job Loss from Formal
    delta_I = Pi_s[I, U],   # Job Loss from Informal
    pi_F    = 0.30,         # Calibrated: Formal Offer Prob   | Employed
    pi_I    = 0.20,         # Calibrated: Informal Offer Prob | Employed
    pi_UF   = 0.45,         # Calibrated: Formal Offer Prob   | Unemployed
    pi_UI   = 0.70,         # Calibrated: Informal Offer Prob | Unemployed
    sig     = 0.5,          # SMM: Smoothness of Tastes       -> xxx

    # --- Sector Productivities ---
    mu_I    = -0.5,        # SMM: Informal Productivity    -> Median Wage Gap (q50)
    sigma_F = 0.30,        # SMM: Formal Volatility        -> Formal Wage Spread (q25, q75)
    sigma_I = 0.40,        # SMM: Informal Volatility      -> Informal Wage Spread (q25, q75)
    nT      = 5,

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,         # SMM: Persistence of Productivity     -> Persistence of Income
    sd_e  = 0.50,          # SMM: Sd of Persistent Productivity   -> Wage Spread (q25, q50, q75)
    nE    = 15,
    amin  = 0.0,
    amax  = 100.0,
    nA    = 100,

    # --- Monetary ---
    phi   = 1.5,          # Taylor Rule Coefficient
    rstar = 0.0125,       # Real Interest Rate (5% Annual)
    pi    = 0.0,          # Normalized: Inflation Deviation (Steady State)

    # --- Government ---
    tau_l = 0.13,                       # Labor Tax = 13% of Wage
    BF_w  = pnad['BF_w'],               # BF Payment / Wage Bill
    B_gdp = pnad['B_gdp'] * 4,          # Debt / GDP (quarterly)
    phi_B = 1.0125 - 0.5 ** (1/20),     # Response to Pay off Debt: 50% Repaid in 5y

    # --- Bolsa Familia Rule ---
    lambda_BF = 0.30,             # Calibrated: Entry Prob, if Eligible              -> BF_U
    ybar      = pnad['ybar'],     # Calibrated: Means-Test Threshold (% of E[y|F])   -> BF_I
    rho_F     = 0.30,             # Calibrated: Formal Audit Probability             -> BF_F
    rho_I     = 1/8,              # Informal Audit Probability (2y)
    sig_BF    = 0.05,             # Smoothness of the Means Test

    # --- Firms ---
    w        = 1.0,      # Normalized
    mu       = 1.11,     # Price Markup
    mu_w     = 1.11,     # Wage Markup
    kappa    = 0.10,     # Price PC Slope
    kappa_w  = 0.10,     # Wage PC Slope
)

# SMM Estimates
smm_paths = dict(g='output/smm_global.csv', l='output/smm_polish.csv')
guess     = dict(calibration)     # Hand-set Start for SMM

smm_est = {k: v for f in smm_paths.values() if os.path.exists(f) and USE_SMM
           for k, v in pd.read_csv(f, index_col=0)['value'].items()}
calibration |= smm_est
if smm_est:
    print("SMM: " + "  ".join(f"{k}={v:.4f}" for k, v in smm_est.items()))



# ---------------------------------------------------------------------------
# 2. Internal Calibration
# Targeted Moments
mom_risk   = ['ac4']                                             # Idiosyncratic Risk
mom_wealth = ['gini', 'mpc']                                     # Wealth Distribution
mom_wage   = [f'q{q}_{s}' for s in 'FI' for q in qs]             # Wage Distribution
mom_wage_t = [f'q{q}_{s}' for s in 'FI' for q in (25, 50, 75)]   # q10 and q90 untargeted

mom_smm    = [*mom_wage_t, *mom_wealth, *mom_risk]               # Targeted Moments
mom_fix    = [*flows, 'BF', 'BF_F', 'BF_I', 'BF_U', 'BF_w']      # Matched by Construction


# Moments in Data
mom_data = dict(
    # --- Targeted (SMM) ---
    **flows,                            # Sector Flows
    **wid,                              # Wealth Distribution (gini, htm, mpc)
    **{k: pnad[k] for k in mom_wage},   # Wage Distribution (q10, q25, q50, q75, q90)
    ac4     = pnad['ac4'],              # Corr(log y_t, log y_t+4 | F)

    # --- Untargeted ---
    ac1     = pnad['ac1'],              # Corr(log y_t, log y_t+1 | F)
    xi      = pnad['xi'],               # E[y_I] / E[y_F]
    h_ratio = pnad['h_ratio'],          # E[h_I] / E[h_F]
    BF      = pnad['BF'],               # Bolsa Familia Coverage
    BF_F    = pnad['BF_F'],             # P(BF | F)
    BF_I    = pnad['BF_I'],             # P(BF | I)
    BF_U    = pnad['BF_U'],             # P(BF | U)
    BF_w    = pnad['BF_w'],             # Total Spending / Wage Bill
    Tr_yF   = pnad['Tr_yF'],            # Average Transfer / E[y_F]
    y35     = pnad['y35'],              # 35th Percentile of Labor Income / E[y_F], vs ybar
)

# Design-based SE for the weight matrix
mom_se = {k: pnad_se[k] if k in pnad_se else 0.0 for k in mom_smm}


# ---- Parameters -----------------------------------------------------------
# SMM Space:      name -> (lower, upper, transform)
smm_space = {
    'mu_I'    : (-2.5,  0.5,   'lin'),     # log(theta_s) ~ N(mu_s, sigma_s^2)
    'sigma_F' : ( 0.05, 1.50,  'log'),
    'sigma_I' : ( 0.05, 1.80,  'log'),
    'rho_e'   : ( 0.80, 0.999, 'logit'),   # log(e_{t+1}) = rho_e log(e_t) + epsilon_t
    'sd_e'    : ( 0.10, 1.20,  'log'),     # e_t ~ N(0, sd_e^2)
    'dbeta'   : ( 0.00, 0.40,  'logit'),   # beta spread      -> HtM and MPC
    'omega_I' : ( 0.05, 0.95,  'logit'),   # Impatient Mass   -> Wealth-Gini
    # 'sig'     : ( 0.10, 2.00,  'log'),     # Taste Dispersion
}



# ---------------------------------------------------------------------------
# 3. Distributional Statistics
def gini_coefficient(values, weights=None):
    # Weighted Gini of raw (value, mass) data.
    idx = np.argsort(values)
    values = values[idx]
    weights = np.ones_like(values) if weights is None else weights[idx]
    pop  = np.concatenate([[0], np.cumsum(weights) / np.sum(weights)])
    wlth = np.concatenate([[0], np.cumsum(weights * values) / np.sum(weights * values)])
    return 1.0 - np.sum((pop[1:] - pop[:-1]) * (wlth[1:] + wlth[:-1]))


def _wquantile(x, wgt, qs):
    # Quantiles of a discrete distribution, CDF interpolated between the nodes.
    i = np.argsort(x); x, wgt = x[i], wgt[i]
    cdf = (np.cumsum(wgt) - 0.5 * wgt) / wgt.sum()
    return np.interp(np.asarray(qs) / 100, cdf, x)


def gini_from_lorenz(pop, share):
    return 1.0 - np.sum(np.diff(pop) * (share[1:] + share[:-1]))


def top_share(pop, share, top):
    # Value share held by the richest `top` fraction.
    return 1.0 - np.interp(1 - top, pop, share)



# ---------------------------------------------------------------------------
# 4. Model Moments
def _lag_corr(d, Pi, x, h, mfrom, mto):
    # Corr(x_t, x_{t+h}) on a subsample {mfrom} at t and {mto} at t+h.
    J = (d[:, None] * np.linalg.matrix_power(Pi, h))[np.ix_(mfrom, mto)]
    if J.sum() < 1e-12: return np.nan
    J = J / J.sum()
    a, b = x[mfrom], x[mto]
    pa, pb = J.sum(1), J.sum(0)
    ma, mb = pa @ a, pb @ b
    va, vb = pa @ (a - ma) ** 2, pb @ (b - mb) ** 2
    return float((a @ J @ b - ma * mb) / np.sqrt(max(va * vb, 1e-18)))


def model_moments(ss, lags=(1, 4)):
    g = lambda k: float(ss[k])
    sF, sI = g('F'), g('I')
    m = {'share_F': sF, 'share_I': sI, 'share_U': g('U')}

    # Sector Flows
    hhi = ss.internals['household']
    P = hhi['P']
    m.update({n: P[i, j] for n, (i, j) in sectors.items()})

    # Hours and Earnings
    m['xi']      = (g('N_I') / sI) / (g('N_F') / sF)
    m['h_ratio'] = (g('H_I') / sI) / g('h_F')

    # Wealth Distribution, over the asset grid
    a, a_d = hhi['a_grid'], hhi['D'].sum(0)
    pop    = np.r_[0, np.cumsum(a_d)]
    share  = np.r_[0, np.cumsum(a_d * a) / (a_d @ a)]

    m['A_gdp'] = g('A') / (4 * (g('Y') + g('Y_I')))

    # HtM Proxy: Liquid Wealth under a month of mean formal earnings
    m['htm']   = float(a_d[a <= g('N_F') / g('F') / 3].sum())
    m['gini']  = gini_from_lorenz(pop, share)
    m['mpc']   = g('MPC_Y') / g('INC_L')       # Weighted by Labor Income
    m['bot50'] = 1 - top_share(pop, share, 0.50)
    m['top10'] = top_share(pop, share, 0.10)
    m['top1']  = top_share(pop, share, 0.01)


    # Wage Distribution, in logs and net of E[y|F]
    d = hhi['D'].sum(1); d = d / d.sum()
    logy = (hhi['log_y_f'] + hhi['log_y_i'])[:, 0]

    blk = d.size // 3
    dw  = d[:blk]
    ref = np.log(dw @ np.exp(logy[:blk]) / dw.sum())   # E[y|F]
    q_F = _wquantile(logy[:blk] - ref, dw, qs)
    q_I = _wquantile(logy[blk:2*blk] - ref, d[blk:2*blk], qs)
    m |= {f'q{q}_F': v for q, v in zip(qs, q_F)}
    m |= {f'q{q}_I': v for q, v in zip(qs, q_I)}

    # Bolsa Familia: Coverage and Transfers
    m['BF']    = g('BF')
    m['BF_F']  = g('BF_F_LF') / sF
    m['BF_I']  = g('BF_I_LF') / sI
    m['BF_U']  = g('BF_U_LF') / g('U')
    m['BF_w']  = g('Tr') * g('BF') / (g('w') * (g('N_F') + g('N_I')))   # / Wage Bill
    m['Tr_yF'] = g('Tr') / np.exp(ref)

    # Persistence, over the formals that stay formal
    mF = np.asarray(hhi['f'])[:, 0] > 0.5
    for h in lags: m[f'ac{h}'] = _lag_corr(d, ss['Pi'], logy, h, mF, mF)
    return m
