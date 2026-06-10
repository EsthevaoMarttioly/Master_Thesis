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
    beta_high = 0.98,    # Calibrated: Patient's Discount Factor
    dbeta     = 0.09,    # Difference: beta_high - beta_low
    omega_I   = 0.50,    # Share of Impatient Agents
    q         = 0.1,     # Prob of Redrawing beta Type (Generation = 25y)

    # --- Labor market ---
    p_fi = 0.11,     # Formal     -> Informal      =>    F = 50.3%
    p_if = 0.12,     # Informal   -> Formal        =>    I = 44.7%
    p_iu = 0.02,     # Informal   -> Unemployed    =>    U =  5.0%
    p_ui = 0.11,     # Unemployed -> Informal
    p_uf = 0.05,     # Unemployed -> Formal

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.85,
    nE    = 15,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Aggregate / Prices (steady-state) ---
    Y = 1.0,     # Output (normalized)
    Z = 1.0,     # Calibrated: Productivity

    # --- Government ---
    tau_l = 0.27,    # Labor Tax = 27% of Wage Bill
    y_bar = 0.60,    # Eligibility Threshold for BF
    Tr    = 0.17,    # BF Transfer Size
    B     = 3.2,     # Debt/GDP = 80% (annual)
    G     = 0.2,     # Calibrated: Government Spending

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.01,    # Real Interest Rate (4% annual)
    pi    = 0.0,     # Inflation Deviation in SS = 0% annual

    # --- Firms ---
    mu       = 1.11,   # Price Markup
    kappa    = 0.10,   # Price PC Slope
    kappa_w  = 0.10,   # Wage PC Slope
    xi       = 0.65,   # Informal Wage Gap
)


calibration['r'] = calibration['rstar']


# Implicit variables in steady state
gamma = calibration['p_iu'] / (calibration['p_uf'] + calibration['p_ui'])
alpha = (calibration['p_if'] * (calibration['p_uf'] + calibration['p_ui']) +
          calibration['p_iu'] * calibration['p_uf']) / calibration['p_fi'] / (calibration['p_uf'] + calibration['p_ui'])

implicit_F, implicit_I, implicit_U = alpha / (1 + gamma + alpha), 1 / (1 + gamma + alpha), gamma / (1 + gamma + alpha)

round(implicit_F, 4), round(implicit_I, 4), round(implicit_U, 4)