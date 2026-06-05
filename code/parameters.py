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
    psi    = 1.0,   # disutility of labor
    varphi = 0.5,   # Frisch elasticity

    # --- Discount Factor ---
    beta_high = 0.98,  # patient's discount factor (calibrated)
    dbeta     = 0.06,  # difference between patient and impatient
    omega_I   = 0.25,  # share of impatient agents
    q         = 0.1,   # prob of redrawing beta type (generation = 25y)

    # --- Labor market ---
    eta_s    = 3.0,   # sensitivity of sector switching
    lambda_s = 0.25,  # speed of sector switching
    p_fi = 0.10,      # formal-informal transition (calibrated)
    p_if = 0.10,      # informal-formal transition (calibrated)
    p_iu = 0.05,      # informal-unemployed transition
    p_ui = 0.10,      # unemployed-informal transition
    p_uf = 0.05,      # unemployed-formal transition

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.85,
    nE    = 15,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Aggregate / Prices (steady-state) ---
    Y     = 1.0,     # Output (normalized)
    Y_I   = 1.0,     # Informal Output (normalized)
    Z     = 1.0,     # Productivity (calibrated)
    pi    = 0.0,     # inflation = 0
    rstar = 0.03,    # real interest rate (12% annual)

    # --- Government ---
    tau_l = 0.27,    # labor tax = 25% of GDP
    y_bar = 0.60,    # eligibility threshold for BF
    Tr    = 0.17,    # BF Transfer
    B     = 1.2,     # debt/GDP = 120% (annual)
    G     = 0.2,     # government spending/GDP = 20%

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation

    # --- Firms ---
    mu       = 1.11,   # price markup
    kappa    = 0.10,   # PC Slope
    kappa_w  = 0.10,   # Wage PC Slope
    xi       = 0.60,   # informal wage gap
    phi_out  = 0.5,    # weight of outside option in NKWPC
)


calibration['r'] = calibration['rstar']


