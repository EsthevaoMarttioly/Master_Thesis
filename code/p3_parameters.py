#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Define the parameters of the model,
# including calibration values and unknowns to be estimated.
# ---------------------------------------------------------------------------
#=

import pandas as pd

pnad = pd.read_csv('data/final/pnad_calibration.csv').loc[0]
Pi_s = pd.read_csv('data/final/pnad_transition_matrix.csv', index_col=0)

# Calibration values
calibration = dict(
    # --- Household Preferences ---
    eis    = 0.5,        # EIS = gamma = 0.5 (CRRA sigma = 2)
    psi    = 0.2,        # Calibrated: Disutility of Labor
    varphi = 0.2,        # Frisch Elasticity                      --- attention!
    h_F    = 1.0,        # Normalized: Formal Worked Hours

    # --- Discount Factor ---
    beta_high = 0.98,    # Calibrated: Patient's Discount Factor
    dbeta     = 0.12,    # Difference: beta_high - beta_low  --- attention!
    omega_I   = 0.75,    # Share of Impatient Agents         --- attention!
    q         = 0.1,     # Prob of Redrawing beta Type (Generation = 25y)

    # --- Labor market ---
    F_ss    = pnad['F'],             # Target: Formal Sector in Steady-State
    I_ss    = pnad['I'],             # Target: Informal Sector in Steady-State
    delta_F = Pi_s['U']['F'],        # Job-Loss Probability to Formal
    delta_I = Pi_s['U']['I'],        # Job-Loss Probability to Informal
    pi_F    = 0.20,                  # Calibrated: Formal Offer Probability
    pi_I    = 0.25,                  # Calibrated: Informal Offer Probability
    sig     = 0.25,                  # Smoothness of Max (sigma -> 0)

    # --- Sector Productivities ---
    mu_F    = pnad['mu_F'],      # Formal Average Productivity
    mu_I    = pnad['mu_I'],      # Informal Average Productivity   --- calibrated to y_I/y_F = xi
    sigma_F = pnad['sd_F'],
    sigma_I = pnad['sd_I'],
    nT      = 5,

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.7,
    nE    = 15,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Government ---
    tau_l  = 0.2,      # Labor Tax = 20% of Wage Bill       --- attention!
    Tr     = 0.3,      # Tr/w = R$ 600 / R$ 4294 = 0.14     --- attention!
    y_bar  = 0.2,      # Eligibility Threshold for BF       --- attention!
    B      = 3.2,      # Debt/GDP = 80% (annual)
    tau    = 0.1,      # Calibrated: Transfers
    phi_B  = 0.2,      # Transfer response to debt          --- attention!

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.01,    # Real Interest Rate (4% annual)
    pi    = 0.0,     # Inflation Deviation in SS = 0% annual

    # --- Firms ---
    Y        = 1.0,                      # Normalized: Output
    Z        = 1 / Pi_s['F']['P_ss'],    # Calibrated: Productivity
    mu       = 1.11,     # Price Markup
    mu_w     = 1.11,     # Wage Markup
    kappa    = 0.025,    # Price PC Slope
    kappa_w  = 0.025,    # Wage PC Slope
)


# ---------------------------------------------------------------------------
# Calibrated Values
calibration = {**calibration, 'B_ss': calibration['B'], 'tau_ss': calibration['tau']}
unknowns    = {k: calibration[k] for k in ['beta_high', 'Z', 'tau', 'psi', 'tau_ss']}


# Target Equations
targets = {
    'asset_mkt'   : 0,    # adjust beta_high to A = B
    'labor_mkt'   : 0,    # adjust Z to L = Y/Z = N_F
    'gov_budget'  : 0,    # adjust tau to balance govt budget
    # 'goods_mkt'   : 0,    # untargeted - Walras' Law
    'wage_nkpc'   : 0,    # adjust psi to set h_F = 1
    'debt_rule'   : 0     # adjust tau_ss = tau
}



