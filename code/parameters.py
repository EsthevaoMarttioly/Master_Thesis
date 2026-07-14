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
    psi    = 0.4,   # Calibrated: Disutility of Labor
    varphi = 0.15,  # Frisch Elasticity                      --- attention!
    h_F    = 1.0,   # Normalized: Formal Worked Hours

    # --- Discount Factor ---
    beta_high = 0.98,    # Calibrated: Patient's Discount Factor
    dbeta     = 0.12,    # Difference: beta_high - beta_low  --- attention!
    omega_I   = 0.50,    # Share of Impatient Agents         --- attention!
    q         = 0.1,     # Prob of Redrawing beta Type (Generation = 25y)

    # --- Labor market ---                --- PNAD
    delta_F = 0.01,      # Job-Loss Probability to Formal
    delta_I = 0.05,      # Job-Loss Probability to Informal
    pi_F    = 0.15,      # Formal Offer Probability
    pi_I    = 0.30,      # Informal Offer Probability
    sig     = 0.25,      # Smoothness of Max (sigma -> 0)
    
    # --- Sector Productivities ---       --- PNAD
    mu_F    = 0.00,      # Formal Average Productivity
    mu_I    = -0.6,      # Informal Average Productivity
    sigma_F = 0.04,
    sigma_I = 0.10,
    nT      = 3,

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.85,
    nE    = 15,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Government ---
    tau_l = 0.2,      # Labor Tax = 20% of Wage Bill
    Tr    = 0.2,      # Tr/w = R$ 600 / R$ 4294 = 0.14     --- attention!
    y_bar = 0.2,      # Eligibility Threshold for BF       --- attention!
    B     = 3.2,      # Debt/GDP = 80% (annual)
    tau   = 0.1,      # Calibrated: Transfers

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.01,    # Real Interest Rate (4% annual)
    pi    = 0.0,     # Inflation Deviation in SS = 0% annual

    # --- Firms ---
    Y        = 1.0,    # Normalized: Output
    Z        = 1.6,    # Calibrated: Productivity
    xi       = 0.64,   # Informal Wage Gap
    mu       = 1.11,   # Price Markup
    mu_w     = 1.11,   # Wage Markup
    kappa    = 0.025,  # Price PC Slope
    kappa_w  = 0.025,  # Wage PC Slope
)


# ---------------------------------------------------------------------------
# Values to be calibrated
unknowns = {k: calibration[k] for k in ['beta_high', 'Z', 'tau', 'psi']}


# Target Equations
targets = {
    'asset_mkt'   : 0,    # adjust beta_high to A = B
    'labor_mkt'   : 0,    # adjust Z to L = Y/Z = N_F
    'gov_budget'  : 0,    # adjust tau to balance govt budget
    # 'goods_mkt'   : 0,    # untargeted - Walras' Law
    'wage_nkpc'   : 0,    # adjust psi to set h_F = 1
}

