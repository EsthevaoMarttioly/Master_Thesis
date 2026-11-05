#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Define the external and internal parameters and the SMM estimator.
# ---------------------------------------------------------------------------
#=

import os
import numpy as np
import pandas as pd


# ---- Data and Helpers -----------------------------------------------------
USE_SMM = True          # True: Use SMM estimates if available
F, I, U = 0, 1, 2

pnad, pnad_se = [r for _, r in pd.read_csv('data/final/pnad_calibration.csv').iterrows()]
Pi_s  = pd.read_csv('data/final/pnad_transition_matrix.csv', index_col=0).iloc[:3].values
alpha = np.array(pnad[['F', 'I', 'U']])


# ---------------------------------------------------------------------------
# 1. External Calibration
rstar     = 1.05 ** (1/4) - 1        # Real Interest Rate (5% Annual)
epsilon_w = 10.1
beta_w    = 0.794 * 0.766 + (1 - 0.794) * 0.979       # Avg beta =~ 0.81
mu_w      = epsilon_w / (epsilon_w - 1)
kappa_w   = (1 - 0.746) * (1 - beta_w * 0.746) / 0.746 / (1 + epsilon_w * 0.5)


calibration = dict(
    # --- Household Preferences ---
    eis    = 0.5,                 # EIS = gamma = 0.5 (CRRA sigma = 2)
    varphi = 0.5,                 # Frisch Elasticity
    h_F    = 1.0,                 # Normalized: Formal Worked Hours
    y_u    = 0.02,                # Subsistence Income, or Home Production

    # --- Discount Factor ---
    dbeta     = 0.12,       # SMM: beta_high - beta_low           -> MPC
    omega_I   = 0.50,       # SMM: Share of Impatient Agents      -> Gini
    q         = 0.01,       # Prob of Redrawing beta Type (Generation = 25y)

    # --- Labor market ---
    delta_F = Pi_s[F, U],   # Job Loss from Formal
    delta_I = Pi_s[I, U],   # Job Loss from Informal
    pi_F    = 0.30,         # Calibrated: Formal Offer Prob   | Employed
    pi_I    = 0.40,         # Calibrated: Informal Offer Prob | Employed
    pi_UF   = 0.40,         # Calibrated: Formal Offer Prob   | Unemployed
    pi_UI   = 0.50,         # Calibrated: Informal Offer Prob | Unemployed
    sig     = 0.70,         # Taste Dispersion

    # --- Sector Productivities ---
    mu_I    = 0.0,         # SMM: Informal Productivity    -> Average Wage Gap
    sigma_F = 0.40,        # SMM: Formal Volatility        -> Formal Wage Spread
    sigma_I = 0.30,        # SMM: Informal Volatility      -> Informal Wage Spread
    nT      = 7,

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,         # SMM: Persistence of Productivity     -> Persistence of Income
    sd_e  = 0.60,          # SMM: Sd of Persistent Productivity   -> Wage Spread
    nE    = 15,
    amin  = 0.0,
    amax  = 100.0,
    nA    = 100,

    # --- Monetary ---
    phi   = 1.5,          # Taylor Rule Coefficient
    rstar = rstar,        # Real Interest Rate (5% Annual)
    pi    = 0.0,          # Normalized: Inflation Deviation (Steady State)

    # --- Government ---
    tau_l  = 0.13,                         # Labor Tax = 13% of Wage
    tau_d  = 0.34,                         # Profit Tax: IRPJ + CSLL
    tau_c  = 0.22,                         # Consumption Tax: 13.9% of GDP / 64% C
    BF_w   = pnad['BF_w'],                 # BF Payment / Wage Bill
    B_gdp  = 0.56,                         # Household Assets / GDP (ANBIMA)
    pe_gdp = 0.08,                         # Equity / GDP           (ANBIMA: 12.9% of A)
    phi_B  = 1 + rstar - 0.5 ** (1/20),    # Response to pay off Debt: 50% Repaid in 5y

    # --- Bolsa Familia Rule ---
    lambda_BF = 0.20,          # Calibrated: Entry Prob, if Eligible              -> BF_U
    ybar      = 0.35,          # Calibrated: Means-Test Threshold (% of E[y|F])   -> BF_I
    rho_F     = 0.25,          # Calibrated: Formal Audit Probability             -> BF_F
    rho_I     = 1/8,           # Informal Audit Probability (2y)
    sig_BF    = 0.05,          # Smoothness of the Means Test    (sig -> 0 = hard max)

    # --- Firms ---
    w        = 1.0,             # Normalized
    mu       = mu_w,            # Price Markup
    mu_w     = mu_w,            # Wage Markup
    kappa    = 0.10,            # Price PC Slope
    kappa_w  = kappa_w,         # Formal Wage PC Slope
    kappa_wI = 3 * kappa_w,     # Informal Wage PC Slope
)


# ---------------------------------------------------------------------------
# 2. Internal Calibration
# Parameters                                name -> (lower, upper, transform)
smm_space = {
    'mu_I'    : (-1.0,  0.3,   'lin'),      # log E[exp(theta_I)] - log E[exp(theta_F)]
    'sigma_F' : ( 0.05, 1.00,  'log'),
    'sigma_I' : ( 0.05, 1.20,  'log'),
    'rho_e'   : ( 0.70, 0.999, 'logit'),    # log(e_{t+1}) = rho_e log(e_t) + epsilon_t ~ N(0, sd_e^2)
    'sd_e'    : ( 0.10, 1.20,  'log'),
    'dbeta'   : ( 0.00, 0.40,  'logit'),    # beta spread           ->    HtM and MPC
    'omega_I' : ( 0.05, 0.95,  'logit'),    # share of impatient    ->    Gini
}


# Moments
mom_wage   = ['xi', 'sd_F', 'sd_I', 'dw_FI', 'dw_IF', 'ac4']      # Wage Distribution
mom_wealth = ['gini', 'mpc', 'htm']                               # Wealth Distribution

mom_smm    = ['FI', 'IF', *mom_wage, *mom_wealth]                 # Targeted Moments
pre_block  = {'gap'   : ['mu_I'],                                 # dw_FI, dw_IF
              'wage'  : ['sigma_F', 'sigma_I', 'sd_e', 'rho_e'],  # xi, sd, ac4
              'wealth': ['dbeta', 'omega_I']}                     # gini, mpc, htm


# Untargeted Moments
qs        = (25, 50, 75)
mom_quant = [f'q{q}_{s}' for s in 'FI' for q in qs]               # Wage (quantiles)
mom_fix   = ['FU', 'IU', 'UF', 'UI', 'BF',
             'BF_F', 'BF_I', 'BF_U', 'BF_w']                      # Matched by Construction
mom_pnad  = ['ac1', 'h_ratio', 'BF_w', 'Tr_yF', 'BF']             # Moments from PNAD



# ---- Helpers --------------------------------------------------------------
# SMM Estimates
smm_dir  = 'output/smm/'
smm_path = {'s': smm_dir + 'smm_starts.csv',     # screen():     the Sobol starts
            'm': smm_dir + 'smm_multi.csv',      # collect():    best of the starts
            'f': smm_dir + 'smm_final.csv'}      # polish():     the estimate

smm_est = dict(pd.read_csv(smm_path['f'], index_col=0)['value'].items()
               if USE_SMM and os.path.exists(smm_path['f']) else [])
calibration |= smm_est


# Arrival Rates and Flows Calibration
pi_calib = {'pi_F': (I, F), 'pi_I': (F, I), 'pi_UF': (U, F), 'pi_UI': (U, I)}
sectors  = {'FI': (F, I), 'FU': (F, U), 'IF': (I, F),
            'IU': (I, U), 'UF': (U, F), 'UI': (U, I)}
flows    = {n: Pi_s[i, j] for n, (i, j) in sectors.items()}


# BF Calibration and Initial Guesses
bf_calib = {'lambda_BF': ('BF_U_LF', 'U', 1),      # (moment, sector, sign)
            'ybar'     : ('BF_I_LF', 'I', 1),
            'rho_F'    : ('BF_F_LF', 'F', -1)}
unknowns = dict(beta_high = 0.98, psi = 1.1, L = 0.6, G = 0.1,
                Tr = 0.2, B = 4.0, h_I = 0.9, Phi = 0.0)
update   = [*pi_calib, *bf_calib, *unknowns, 'w_I', 'tau_css', 'B_ss']      # w_I = w, calibrated in the SS
cf_keep  = [*update, 'h_F', 'tau_c']                                       # a counterfactual holds these



# ---- Moments: Data --------------------------------------------------------
# Moments in Data
wid = dict(gini  = 0.82,             # 2025 Global Wealth Report, UBS
           htm   = 0.35,             # Costa-Junior et al. (2025)
           mpc   = 0.20,             # Income-Weighted Quarterly MPC: Auclert et al. (2025)
           top1  = 0.395,            # Brazilian Wealth Shares (WID.world, 2024)
           top10 = 0.719,
           bot50 = 0.02,
)


mom_data = dict(
    # --- Targeted (SMM) ---
    **flows,                                  # Sector Flows
    **{k: pnad[k] for k in mom_wage},         # Wage Distribution
    **{k: pnad[k] for k in mom_quant},        # Wage Quantiles (untargeted)
    **wid,                                    # Wealth Distribution

    # --- Untargeted ---
    **{k: pnad[k] for k in mom_pnad},
    **{f'BF_{s}': pnad[f'BF_{s}'] for s in 'FIU'},            # P(BF | s)
    **{f'cut_{s}': pnad[f'cut_{s}'] for s in ('FI', 'IF')},   # P[dlog w < 0]
)


# Design-based SE for the weight matrix
mom_se = {k: pnad_se[k] if k in pnad_se else 0.0 for k in mom_smm}
mom_fl = {k: 0.25 * abs(flows[k]) for k in ['FI', 'IF']}      # Flows: 25% tolerance



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


def _lag_flow(d, Pi, x, mfrom, mto):
    # E[x_{t+1} - x_t] and P(x falls) over the flow {mfrom} -> {mto}, one quarter.
    J = (d[:, None] * Pi)[np.ix_(mfrom, mto)]
    if J.sum() < 1e-12: return np.nan, np.nan
    J  = J / J.sum()
    dx = x[mto][None, :] - x[mfrom][:, None]
    return float((J * dx).sum()), float(J[dx < 0].sum())


def model_moments(ss, lags=(1, 4)):
    g = lambda k: float(ss[k])
    sF, sI = g('F'), g('I')
    m = {'share_F': sF, 'share_I': sI, 'share_U': g('U')}

    # Sector Flows
    hhi = ss.internals['household']
    P = hhi['P']
    m.update({n: P[i, j] for n, (i, j) in sectors.items()})

    # Hours and Wages
    m['h_ratio'] = g('h_I') / g('h_F')
    m['xi']      = (g('w_I') * g('N_I') / sI) / (g('w') * g('N_F') / sF) / m['h_ratio']

    # Wealth Distribution, over the asset grid
    a, a_d = hhi['a_grid'], hhi['D'].sum(0)
    pop    = np.r_[0, np.cumsum(a_d)]
    share  = np.r_[0, np.cumsum(a_d * a) / (a_d @ a)]


    # HtM Proxy: Liquid Wealth under a month of mean formal earnings
    m['A_gdp'] = g('A') / (4 * (g('Y') + g('Y_I')))
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
    logw = logy - np.r_[np.full(blk, np.log(g('h_F'))),
                        np.full(blk, np.log(g('h_I'))), np.zeros(d.size - 2 * blk)]
    q_F = _wquantile(logw[:blk] - ref, dw, qs)
    q_I = _wquantile(logw[blk:2*blk] - ref, d[blk:2*blk], qs)
    m |= {f'q{q}_F': v for q, v in zip(qs, q_F)}
    m |= {f'q{q}_I': v for q, v in zip(qs, q_I)}

    sd = lambda x, w: float(np.sqrt(w @ (x - w @ x / w.sum()) ** 2 / w.sum()))
    m['sd_F'] = sd(logw[:blk], dw)
    m['sd_I'] = sd(logw[blk:2*blk], d[blk:2*blk])

    # Bolsa Familia: Coverage and Transfers
    m['BF']    = g('BF')
    m['BF_F']  = g('BF_F_LF') / sF
    m['BF_I']  = g('BF_I_LF') / sI
    m['BF_U']  = g('BF_U_LF') / g('U')
    m['BF_w']  = g('Tr') * g('BF') / (g('w') * (g('N_F') + g('N_I')))   # / Wage Bill
    m['Tr_yF'] = g('Tr') / np.exp(ref)
    m['T_BFF'] = 1 / g('rho_F') * 3           # Avg Time for Formals lose the Benefit (months)

    # Persistence over formal stayers, and wage change on a switch
    mF = np.asarray(hhi['f'])[:, 0] > 0.5
    mI = np.asarray(hhi['i'])[:, 0] > 0.5
    for h in lags: m[f'ac{h}'] = _lag_corr(d, ss['Pi'], logw, h, mF, mF)
    m['dw_FI'], m['cut_FI'] = _lag_flow(d, ss['Pi'], logw, mF, mI)
    m['dw_IF'], m['cut_IF'] = _lag_flow(d, ss['Pi'], logw, mI, mF)
    return m
