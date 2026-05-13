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
    eis       = 0.5,   # EIS = gamma = 0.5 (CRRA sigma = 2)
    frisch    = 0.5,   # Labor supply elasticity = 0.5
    vscale    = 1.0,   # scale of labor disutility

    # --- Discount Factor ---
    dbeta     = 0.06,  # difference between patient and impatient
    omega_I   = 0.25,  # share of impatient agents
    q         = 0.1,   # prob of redrawing beta type (generation = 25y)

    # --- Labor market ---
    f     = 0.30,     # job-finding probability
    s     = 0.02,     # job-loss probability         =>  U*+V* = s/(s+f) = 6.25%
    Tb    = 1.5,      # benefit duration (quarters)  =>  Tb >= 1/(1-f)

    # --- Productivity and Asset Grid ---
    rho_e = 0.966,
    sd_e  = 0.5,
    nE    = 7,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,

    # --- Aggregate / Prices (steady-state) ---
    Y     = 1.0,     # Output (normalized)
    pi    = 0.0,     # inflation = 0
    r     = 0.005,   # real interest rate (2% annual)

    # --- Government ---
    # tau   = 0.25,    # labor tax = 25% of GDP
    b     = 0.3,     # unemployment benefit
    Tr    = 0.1,     # safety-net transfer to needy
    B     = 1.2,     # debt/GDP = 120% (annual)

    # --- Monetary ---
    phi   = 1.5,     # Taylor rule coefficient on inflation

    # --- Firms ---
    mu      = 1.11,    # price/wage markup
    kappa   = 0.1,     # PC Slope
)


calibration['rstar'] = calibration['r']


