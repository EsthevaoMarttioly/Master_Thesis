#=
#---------------------------------------------------------------------------
# DESCRIPTION
# Define the parameters of the model,
# including calibration values and unknowns to be estimated.
#---------------------------------------------------------------------------
#=

# Calibration values
calibration = dict(
    # Households
    eis   = 0.5,       # elasticity of intertemporal substitution
    lambda_I  = 0.25,  # share of impatient agents
    q         = 0.1,   # prob of redrawing beta type (generation = 25y)
    dbeta     = 0.05,  # difference between patient and impatient

    # Labor market
    f     = 0.4,     # job-finding probability
    s     = 0.1,     # separation probability  =>  U_ss = s/(s+f) = 0.2
    Tb    = 2.0,     # unemployment benefit duration (in quarters)

    # Productivity and asset grid
    rho_e = 0.966,
    sd_e  = 0.5,
    nE    = 7,
    amin  = 0.0,
    amax  = 200.0,
    nA    = 500,

    # Prices (SS targets/normalizations)
    Y     = 1.0,     # Output (normalized)
    pi    = 0.0,     # inflation = 0 at SS
    r     = 0.005,   # real interest rate at SS

    # Government
    tau   = 0.25,    # labor tax = 25% of GDP
    b     = 0.2,     # unemployment benefit
    Tr    = 0.1,     # safety-net transfer to needy
    B     = 1.2,     # debt = 120% GPD

    # Monetary
    phi   = 1.5,     # Taylor rule coefficient on inflation

    # Firms
    mu      = 1.11,    # price markup
    kappa   = 0.1,     # price PC slope
    mu_w    = 1.2,     # wage markup
    kappa_w = 0.025,   # wage NKPC slope
)


# Extra parameters derived from calibration
calibration["rstar"] = calibration["r"]
calibration["Z"] = 1 / (1 - calibration["s"]/(calibration["s"]+calibration["f"]))  # unemployment rate at SS


# Unknown values to be estimated, with initial guess
unknowns_ss = dict(
    w    = 0.7,        # real wage - solved by NKPC
    beta_high = 0.97,  # patient's discount factor
)

