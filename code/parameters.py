#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Define the parameters of the model,
# including calibration values and unknowns to be estimated.
# ---------------------------------------------------------------------------
#=

# Calibration values
calibration = dict(
    # Households
    eis       = 0.5,   # elasticity of intertemporal substitution
    lambda_I  = 0.25,  # share of impatient agents
    q         = 0.1,   # prob of redrawing beta type (generation = 25y)
    dbeta     = 0.05,  # difference between patient and impatient
    # Labor market
    f     = 0.30,     # job-finding probability
    s     = 0.02,     # job-loss probability   =>  U*+V* = s/(s+f) = 6.25%
    Tb    = 1.5,     # average benefit duration (quarters)  =>  Tb >= 1/(1-f)
    # Productivity and asset grid
    rho_e = 0.966,
    sd_e  = 0.5,
    nE    = 7,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 200,
    # Prices (SS targets/normalizations)
    Y     = 1.0,     # Output (normalized)
    pi    = 0.0,     # inflation = 0 at SS
    r     = 0.005,   # real interest rate at SS
    # Government
    tau   = 0.25,    # labor tax = 25% of GDP
    b     = 0.3,     # unemployment benefit
    Tr    = 0.1,     # safety-net transfer to needy
    B     = 1.2,     # debt = 120% GPD
    # Monetary
    phi   = 1.5,     # Taylor rule coefficient on inflation
    # Firms
    mu      = 1.11,    # markup
    kappa   = 0.1,     # price PC slope
    kappa_w = 0.1,     # wage NKPC slope
)


# Extra parameters derived from calibration
calibration["rstar"] = calibration["r"]
# calibration["Z"] = 1 / (1 - calibration["s"]/(calibration["s"]+calibration["f"]))  # unemployment rate at SS



