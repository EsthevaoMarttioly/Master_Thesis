#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Define the parameters of the model,
# including calibration values and unknowns to be estimated.
# ---------------------------------------------------------------------------
#=

# Calibration values
calibration = dict(
    # --- Household Preferences ---
    eis    = 0.5,   # EIS = gamma = 0.5 (CRRA sigma = 2)
    psi    = 0.6,   # Disutility of Labor
    varphi = 0.15,  # Frisch Elasticity
    h_F    = 1.0,   # Normalized: Formal Worked Hours

    # --- Discount Factor ---
    beta_high = 0.98,    # Calibrated: Patient's Discount Factor
    dbeta     = 0.12,    # Difference: beta_high - beta_low
    omega_I   = 0.50,    # Share of Impatient Agents
    q         = 0.1,     # Prob of Redrawing beta Type (Generation = 25y)

    # --- Labor market ---
    p_fi = 0.11,     # Formal     -> Informal      =>    F = 50.5%  (data = 50.0%)
    p_if = 0.12,     # Informal   -> Formal        =>    I = 44.0%  (data = 44.2%)
    p_iu = 0.02,     # Informal   -> Unemployed    =>    U =  5.5%  (data =  5.8%)
    p_ui = 0.11,     # Unemployed -> Informal
    p_uf = 0.05,     # Unemployed -> Formal

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.85,
    nE    = 15,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Government ---
    tau_l = 0.27,          # Labor Tax = 27% of Wage Bill
    Tr    = 0.14 * 1.8,    # Tr/w = R$ 600 / R$ 4294 = 0.14
    y_bar = 1.0,           # Eligibility Threshold for BF
    B     = 3.2,           # Debt/GDP = 80% (annual)
    G     = 0.2,           # Calibrated: Government Spending

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.01,    # Real Interest Rate (4% annual)
    pi    = 0.0,     # Inflation Deviation in SS = 0% annual

    # --- Firms ---
    Y        = 1.0,    # Normalized: Output
    Z        = 1.0,    # Calibrated: Productivity
    xi       = 0.64,   # Informal Wage Gap
    mu       = 1.11,   # Price Markup
    mu_w     = 1.11,   # Wage Markup
    kappa    = 0.025,  # Price PC Slope
    kappa_w  = 0.025,  # Wage PC Slope
)



# ---------------------------------------------------------------------------
# Analitical Steady State Solution
gamma = calibration['p_iu'] / (calibration['p_uf'] + calibration['p_ui'])
alpha = (calibration['p_if'] * (calibration['p_uf'] + calibration['p_ui']) +
          calibration['p_iu'] * calibration['p_uf']) /\
              (calibration['p_fi'] * (calibration['p_uf'] + calibration['p_ui']))


calibration_ss = calibration | dict(
    F  = alpha / (1 + gamma + alpha),
    I  = 1 / (1 + gamma + alpha),
    U  = gamma / (1 + gamma + alpha),
    r  = calibration['rstar'],
    A   = calibration['B'],
    Div = (1 - calibration['tau_l']) * (1 - 1/calibration['mu']),
)


calibration_ss = calibration_ss | dict(
    w   = calibration['Y'] / (calibration['mu'] * calibration_ss['F']),
    w_I = calibration['Y'] / (calibration['mu'] * calibration_ss['F']) * calibration['xi'],
    Z   = calibration['Y'] / calibration_ss['F'],
)


calibration_ss.pop('G', None)
calibration_ss.pop('psi', None)
calibration_ss.pop('beta_high', None)



# import numpy as np
# from scipy.stats import norm

## Calibration BF = U* + I* . F(e < e*)
# e_star = (calibration['psi'] ** calibration['varphi'] *\
#           calibration['y_bar']) ** (1/(1+calibration['varphi'])) / calibration_ss['w_I']

# BF_ = calibration_ss['U'] + calibration_ss['I'] *\
#       norm.cdf(np.log(e_star) / calibration['sd_e'] * np.sqrt(1 - calibration['rho_e']**2))
