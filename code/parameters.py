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
    psi    = 1.0,   # Disutility of Labor
    varphi = 0.5,   # Frisch Elasticity

    # --- Discount Factor ---
    beta_high = 0.98,  # Calibrated: Patient's discount factor
    dbeta     = 0.06,  # Difference between patient and impatient
    omega_I   = 0.25,  # Share of impatient agents
    q         = 0.1,   # Prob of redrawing beta type (generation = 25y)

    # --- Labor market ---
    eta_s    = 3.0,   # Sensitivity of sector switching
    lambda_s = 0.25,  # Speed of sector switching
    p_fi = 0.10,      # Calibrated: formal-informal transition
    p_if = 0.10,      # Calibrated: informal-formal transition
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
    Z     = 1.0,     # Calibrated: Productivity

    # --- Government ---
    tau_l = 0.27,    # Labor Tax = 27% of Wage Bill
    y_bar = 0.60,    # Eligibility Threshold for BF
    Tr    = 0.17,    # BF Transfer
    B     = 1.2,     # Debt/GDP = 120% (annual)
    G     = 0.2,     # Calibrated: Government Spending

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.03,    # Real Interest Rate (12% annual)
    pi    = 0.0,     # Inflation = 0

    # --- Firms ---
    mu       = 1.11,   # Price Markup
    kappa    = 0.10,   # Price PC Slope
    kappa_w  = 0.10,   # Wage PC Slope
    xi       = 0.60,   # Informal Wage Gap
    phi_out  = 1.0,    # Weight of Outside Option in NKWPC
)


calibration['r'] = calibration['rstar']


