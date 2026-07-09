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
    psi    = 0.3,   # Calibrated: Disutility of Labor
    varphi = 0.15,  # Frisch Elasticity
    h_F    = 1.0,   # Normalized: Formal Worked Hours

    # --- Discount Factor ---
    beta_high = 0.96,    # Calibrated: Patient's Discount Factor
    dbeta     = 0.12,    # Difference: beta_high - beta_low
    omega_I   = 0.50,    # Share of Impatient Agents
    q         = 0.1,     # Prob of Redrawing beta Type (Generation = 25y)

    # --- Labor market ---
    delta_F = 0.03,      # Job-Loss Probability to Formal
    delta_I = 0.10,      # Job-Loss Probability to Informal
    pi_F    = 0.10,      # Formal Offer Probability
    pi_I    = 0.30,      # Informal Offer Probability
    sig     = 0.10,      # Smoothness of Max (sigma -> 0)
    
    # --- Sector Productivities ---
    mu_F    = 0.00,      # Formal Average Productivity
    mu_I    = -0.20,     # Informal Average Productivity
    sigma_F = 0.20,
    sigma_I = 0.40,
    nT      = 3,

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.85,
    nE    = 7,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Government ---
    tau_l = 0.27,          # Labor Tax = 27% of Wage Bill
    Tr    = 0.14 * 1.8,    # Tr/w = R$ 600 / R$ 4294 = 0.14
    y_bar = 0.6,           # Eligibility Threshold for BF
    B     = 3.2,           # Debt/GDP = 80% (annual)
    G     = 0.1,           # Calibrated: Government Spending

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.01,    # Real Interest Rate (4% annual)
    pi    = 0.0,     # Inflation Deviation in SS = 0% annual

    # --- Firms ---
    Y        = 1.0,    # Normalized: Output
    Z        = 2.0,    # Calibrated: Productivity
    xi       = 0.64,   # Informal Wage Gap
    mu       = 1.11,   # Price Markup
    mu_w     = 1.11,   # Wage Markup
    kappa    = 0.025,  # Price PC Slope
    kappa_w  = 0.025,  # Wage PC Slope
)


# ---------------------------------------------------------------------------
# Values to be calibrated
unknowns = {k: calibration[k] for k in ['beta_high', 'Z', 'G', 'psi']}


# Target Equations
targets = {
    'asset_mkt'   : 0,    # adjust beta_high to A = B
    'labor_mkt'   : 0,    # adjust Z to L = Y/Z = N_F
    'gov_budget'  : 0,    # adjust G to balance govt budget
    # 'goods_mkt'   : 0,    # untargeted - Walras' Law
    'wage_nkpc'   : 0,    # adjust phi to set h_F = 1
}

