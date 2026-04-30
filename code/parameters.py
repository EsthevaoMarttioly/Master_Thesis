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
    eis   = 0.5,     # elasticity of intertemporal substitution
    # Patiency
    lambda_I  = 0.25,    # share of impatient agents
    q         = 0.1,     # prob of redrawing beta type (near-permanent)
    dbeta     = 0.05,    # difference between patient and impatient
    # Labor market
    f     = 0.4,     # job-finding probability
    s     = 0.1,     # separation probability  =>  U_ss = s/(s+f) = 0.2
    # Productivity process (Rouwenhorst)
    rho_e = 0.966,
    sd_e  = 0.5,
    nE    = 7,
    # Asset grid
    amin  = 0.0,
    amax  = 200.0,
    nA    = 500,
    # Prices (SS targets/normalizations)
    Y     = 1.0,     # Output (normalized)
    Z     = 1.25,    # TFP (normalized to L = 0.8)
    pi    = 0.0,     # inflation = 0 at SS
    r     = 0.005,   # real interest rate at SS
    # Government
    tau   = 0.25,    # labor tax = 40% GDP
    Tr    = 0.0,     # lump-sum Transfers = 20% GDP
    B     = 1.2,     # debt = 120% GPD
    # Monetary
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.005,   # SS neutral real rate
    # Firms
    mu    = 1.11,    # price markup
    kappa = 0.1,     # NKPC slope (Rotemberg adjustment cost)
)


# Unknown values to be estimated, with initial guess
unknowns_ss = dict(
    w    = 0.7,     # real wage - solve by NKPC
    beta_high = 0.97,     # patient's discount factor
)


# Y / Z = L = 1 - U
# w = Z/mu