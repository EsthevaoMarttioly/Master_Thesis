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
    delta_F = 0.04,      # Job-Loss Probability to Formal
    delta_I = 0.06,      # Job-Loss Probability to Informal
    pi_F    = 0.10,      # Formal Offer Probability
    pi_I    = 0.30,      # Informal Offer Probability
    sig     = 1e-6,      # Smoothness of max (sigma -> 0)
    
    # --- Sector Productivities ---
    mu_F    = 0.00,      # Formal Average Productivity
    mu_I    = -0.10,     # Informal Average Productivity
    sigma_F = 0.20,
    sigma_I = 0.40,
    nT      = 3,

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
gamma = calibration['delta_I'] / (calibration['pi_F'] + calibration['pi_I'])
alpha = (calibration['pi_F'] * (calibration['pi_F'] + calibration['pi_I']) +
          calibration['delta_I'] * calibration['pi_F']) /\
              (calibration['pi_I'] * (calibration['pi_F'] + calibration['pi_I']))


calibration_ss = calibration | dict(
    F  = alpha / (1 + gamma + alpha),
    I  = 1 / (1 + gamma + alpha),       # Wrong
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
