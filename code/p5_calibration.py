#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Define the external and internal parameters and the SMM estimator.
# ---------------------------------------------------------------------------
#=

import os
import numpy as np
import pandas as pd


# ---- Data -----------------------------------------------------------------
Pi_s          = pd.read_csv('data/final/pnad_transition_matrix.csv', index_col=0).iloc[:3].values
pnad, pnad_se = [r for _, r in pd.read_csv('data/final/pnad_calibration.csv').iterrows()]
alpha         = np.array(pnad[['F', 'I', 'U']])
F, I, U = 0, 1, 2


# ---------------------------------------------------------------------------
# 1. External Calibration
calibration = dict(
    # --- Household Preferences ---
    eis    = 0.5,        # EIS = gamma = 0.5 (CRRA sigma = 2)
    varphi = 0.2,        # Frisch Elasticity                      --- attention!
    h_F    = 1.0,        # Normalized: Formal Worked Hours

    # --- Discount Factor ---
    dbeta     = 0.18,    # SMM: beta_high - beta_low              -> Wealth
    omega_I   = 0.50,    # SMM: Share of Impatient Agents         -> HtM
    q         = 0.01,    # Prob of Redrawing beta Type (Generation = 25y = 100q)

    # --- Labor market ---
    delta_F = Pi_s[F, U],   # Job-Loss Probability from Formal
    delta_I = Pi_s[I, U],   # Job-Loss Probability from Informal
    pi_F    = 0.15,      # Calibrated: Formal Offer Prob   | Employed
    pi_I    = 0.40,      # Calibrated: Informal Offer Prob | Employed
    pi_UF   = 0.20,      # Calibrated: Formal Offer Prob   | Unemployed
    pi_UI   = 0.50,      # Calibrated: Informal Offer Prob | Unemployed
    sig     = 0.50,      # SMM: Smoothness of Tastes     -> E[dLog y | F->I]

    # --- Sector Productivities ---
    mu_I    = -1.00,       # SMM: Informal Productivity    -> Median Wage Gap
    sigma_F = 0.10,        # SMM: Formal Volatility        -> Formal Wage Spread
    sigma_I = 0.30,        # SMM: Informal Volatility      -> Informal Wage Spread
    nT      = 5,

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,         # SMM: Persistence of Productivity     -> PNAD
    sd_e  = 0.85,          # SMM: Sd of Persistent Productivity   -> Wealth
    nE    = 15,
    amin  = 0.0,
    amax  = 100.0,
    nA    = 200,

    # --- Government ---
    tau_l = 0.2,                      # Labor Tax = 20% of Wage         --- attention!
    BF_w  = pnad['BF_w'],             # BF Payment / Wage Bill
    B_gdp = pnad['B_gdp'] * 4,        # Debt / GDP (quarterly)
    phi_B = 1.01 - 0.05 ** (1/40),    # Response to Pay off Debt: 95% Repaid in 10y

    # --- Bolsa Familia Coverage ---
    **{k: pnad[k] for k in ('phi_F', 'ybar_F', 'sig_F',
                            'phi_I', 'ybar_I', 'sig_I')},
    p_U    = pnad['BF_U'],            # Unemployed: Flat Coverage

    # --- Monetary ---
    phi   = 1.5,         # Taylor Rule Coefficient
    rstar = 0.01,        # Real Interest Rate (4% Annual)
    pi    = 0.0,         # Normalized: Inflation Deviation (Steady State)

    # --- Firms ---
    w        = 1.0,      # Normalized
    mu       = 1.11,     # Price Markup
    mu_w     = 1.11,     # Wage Markup
    kappa    = 0.025,    # Price PC Slope
    kappa_w  = 0.025,    # Wage PC Slope
)

# SMM Estimates: the Polish overrides the Global (delete the file to reset)
smm_paths = dict(g='output/smm_global.csv', l='output/smm_polish.csv')
guess     = dict(calibration)     # Hand-set Start for SMM

smm_est = {k: v for f in smm_paths.values() if os.path.exists(f)
           for k, v in pd.read_csv(f, index_col=0)['value'].items()}
calibration |= smm_est
if smm_est:
    print("SMM: " + "  ".join(f"{k}={v:.4f}" for k, v in smm_est.items()))



# ---------------------------------------------------------------------------
# 2. Internal Calibration
# Arrival Rates
pi_calib = {'pi_F': (I, F), 'pi_I': (F, I), 'pi_UF': (U, F), 'pi_UI': (U, I)}
flows    = {'FIU'[i] + 'FIU'[j]: Pi_s[i, j] for i, j in pi_calib.values()}
bf_calib = {f'phi_{s}': (f'BF_{s}_LF', s) for s in 'FI'}


# Brazilian Wealth Shares (WID.world, 2024)
wid = dict(gini  = 0.82,     # 2025 Global Wealth Report, UBS
           top50 = 0.02,
           top10 = 0.719,
           top1  = 0.395,
           htm   = 0.35,     # Hand-to-Mouth Share
           mpc50 = 0.609)    # Bottom 50% MPC

qs = (10, 25, 50, 75, 90)


# Targeted Moments
mom_risk   = ['ac4']                                             # Idiosyncratic Risk
mom_wealth = ['top10', 'htm', 'mpc50']                           # Wealth Distribution
mom_switch = ['dw_FI', 'dw_IF']                                  # Selection of the Switchers
mom_wage   = [f'q{q}_{s}' for s in 'FI' for q in qs]             # Wage Distribution
mom_wage_t = [f'q{q}_{s}' for s in 'FI' for q in (25,50,75)]     # Targeted Wage Distribution
mom_smm    = [*mom_wage_t, *mom_wealth, *mom_risk, *mom_switch]  # Targeted Moments
mom_fix    = [*flows, 'BF', 'BF_F', 'BF_I', 'BF_U', 'BF_w']      # Matched by Construction


mom_data = dict(
    # --- Targeted (SMM) ---
    **flows,                            # Sector Flows
    **{k: pnad[k] for k in mom_wage},   # Wage Distribution
    **wid,                              # Wealth Distribution
    ac4     = pnad['ac4'],              # Corr(log y_t, log y_t+4 | F)
    dw_FI   = pnad['dw_FI'],            # E[dLog y | F -> I]
    dw_IF   = pnad['dw_IF'],            # E[dLog y | I -> F]

    # --- Untargeted ---
    xi      = pnad['xi'],               # E[y_I] / E[y_F]
    h_ratio = pnad['h_ratio'],          # E[h_I] / E[h_F]
    BF      = pnad['BF'],               # Bolsa Familia Coverage
    BF_F    = pnad['BF_F'],             # P(BF | F)
    BF_I    = pnad['BF_I'],             # P(BF | I)
    BF_U    = pnad['BF_U'],             # P(BF | U)
    BF_w    = pnad['BF_w'],             # Total Spending / Wage Bill
    Tr_yF   = pnad['Tr_yF'],            # Average Transfer / E[y_F]
    ac1     = pnad['ac1'],              # Corr(log y_t, log y_t+1 | F)
    mpc     = 0.20,                     # Aggregate MPC: Auclert et al. (2025)
)

# Design-based SE for the weight matrix
mom_se = {k: pnad_se[k] if k in pnad_se else 0.0 for k in mom_smm}


# ---- Parameters -----------------------------------------------------------
# SMM Space:      name -> (lower, upper, transform)
smm_space = {
    'mu_I'    : (-2.5,  0.0,   'lin'),     # log(theta_s) ~ N(mu_s, sigma_s^2)
    'sigma_F' : ( 0.05, 1.50,  'log'),
    'sigma_I' : ( 0.05, 1.80,  'log'),
    'rho_e'   : ( 0.80, 0.999, 'logit'),   # log(e_{t+1}) = rho_e log(e_t) + epsilon_t
    'sd_e'    : ( 0.10, 1.20,  'log'),     # epsilon_t ~ N(0, sd_e^2)
    'dbeta'   : ( 0.00, 0.40,  'logit'),   # beta spread      -> top of the distribution
    'omega_I' : ( 0.05, 0.95,  'logit'),   # impatient mass   -> bottom (HtM)
    'sig'     : ( 0.10, 2.00,  'log'),     # taste dispersion -> dLogY of switchers
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


def gini_from_lorenz(pop, share):
    return 1.0 - np.sum(np.diff(pop) * (share[1:] + share[:-1]))


def top_share(pop, share, top):
    # Value share held by the richest `top` fraction.
    return 1.0 - np.interp(1 - top, pop, share)



# ---------------------------------------------------------------------------
# 4. Model Moments
def _wquantile(x, wgt, qs):
    # Quantiles of a discrete distribution, CDF interpolated between the nodes.
    i = np.argsort(x); x, wgt = x[i], wgt[i]
    cdf = (np.cumsum(wgt) - 0.5 * wgt) / wgt.sum()
    return np.interp(np.asarray(qs) / 100, cdf, x)


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


def _switch_gap(d, Pi, x, mfrom, mto):
    # E[x_{t+1} - x_t] for the switchers from {mfrom} to {mto}.
    J = (d[:, None] * Pi)[np.ix_(mfrom, mto)]
    if J.sum() < 1e-12: return np.nan
    J = J / J.sum()
    return float(J.sum(0) @ x[mto] - J.sum(1) @ x[mfrom])


def model_moments(ss, lags=(1, 4)):
    g = lambda k: float(ss[k])
    sF, sI = g('F'), g('I')
    m = {'share_F': sF, 'share_I': sI, 'share_U': g('U')}

    # Sector Flows
    hhi = ss.internals['household']
    P = hhi['P']
    m.update({n: P[i] for n, i in zip(flows, pi_calib.values())})

    # Hours and Earnings, from the aggregates
    m['h_ratio'] = (g('H_I') / sI) / g('h_F')
    m['xi']      = (g('N_I') / sI) / (g('N_F') / sF)

    # Wealth Distribution, over the asset grid
    a, a_d = hhi['a_grid'], hhi['D'].sum(0)
    pop    = np.r_[0, np.cumsum(a_d)]
    share  = np.r_[0, np.cumsum(a_d * a) / (a_d @ a)]
    m['htm']   = a_d[0]                      # mass at the borrowing constraint
    m['gini']  = gini_from_lorenz(pop, share)
    m['top10'] = top_share(pop, share, 0.10)
    m['top1']  = top_share(pop, share, 0.01)

    # MPC:   1 - da'/dcoh, with dcoh = (1+r) da
    mpc = 1 - np.diff(hhi['a'], axis=1) / ((1 + g('r')) * np.diff(a))
    mpc = np.c_[mpc, mpc[:, -1]]
    bot = slice(0, np.searchsorted(pop[1:], 0.50) + 1)     # bottom 50% of wealth
    m['mpc']   = float((hhi['D'] * mpc).sum())
    m['mpc50'] = float((hhi['D'][:, bot] * mpc[:, bot]).sum() / a_d[bot].sum())

    d = hhi['D'].sum(1); d = d / d.sum()
    logy = (hhi['log_y_f'] + hhi['log_y_i'])[:, 0]

    # Wage Distribution, in logs and net of E[y|F]
    blk = d.size // 3
    ref = np.log(g('w') * g('N_F') / sF)      # E[y|F], as PNAD nets the quantiles
    q_F = _wquantile(logy[:blk] - ref, d[:blk], qs)
    q_I = _wquantile(logy[blk:2*blk] - ref, d[blk:2*blk], qs)
    m |= {f'q{q}_F': v for q, v in zip(qs, q_F)}
    m |= {f'q{q}_I': v for q, v in zip(qs, q_I)}

    # Bolsa Familia: Coverage and Transfers
    m['BF']   = g('BF')
    m['BF_F'] = g('BF_F_LF') / sF
    m['BF_I'] = g('BF_I_LF') / sI
    m['BF_U'] = g('BF_U_LF') / g('U')
    m['BF_w'] = g('Tr') * g('BF') / g('wage_bill')
    m['Tr_yF'] = g('Tr') / np.exp(ref)

    # Persistence
    mF = np.asarray(hhi['f'])[:, 0] > 0.5
    mI = np.asarray(hhi['i'])[:, 0] > 0.5
    Pi = ss['Pi']
    for h in lags: m[f'ac{h}'] = _lag_corr(d, Pi, logy, h, mF, mF)

    # Wage Gain of the Switchers
    m['dw_FI'] = _switch_gap(d, Pi, logy, mF, mI)
    m['dw_IF'] = _switch_gap(d, Pi, logy, mI, mF)
    return m
