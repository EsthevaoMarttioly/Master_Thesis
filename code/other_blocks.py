#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the firms, government and monetary policy blocks of the model,
# as well as the market clearing conditions.
#
# SS-DAG   (for calibration)   +   Dynamics DAG (for IRFs and Jacobians)
#----------------------------------------------------------------------------
#=

# Import Packages
import numpy as np
from sequence_jacobian import simple


#----------------------------------------------------------------------------
# Firm Block:
# 1. Production
@simple
def firm(Y, w, Z):
    L   = Y / Z       # Y = Z * L  =>  L = Y / Z
    Div = Y - w * L
    return L, Div


# 2. SS Phillips Curve
@simple
def nkpc_ss(Z, mu):
    w = Z / mu        # P = mu * W / Z  =>  w = Z / mu
    return w


# 3. Dynamic Phillips Curves
# log(1+pi)   = kappa   * (w/Z - 1/mu) + Y(+1)/Y * log(1+pi(+1)) / (1+r(+1))
# log(1+pi_w) = kappa_w * (w/Z - 1/mu) + log(1+pi_w(+1)) / (1+r(+1))
# 1+pi_w = (1+pi)*w/w(-1)
###    If mu_p != mu_w, both PCs cannot hold simultaneously at the same w_ss

@simple
def price_nkpc(pi, w, Z, Y, r, kappa, mu):
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    return nkpc


@simple
def wage_nkpc(pi, w, Z, r, kappa_w, mu):
    pi_w  = (1 + pi) * w / w(-1) - 1
    nkwpc = (kappa_w * (w / Z - 1 / mu)
             + (1 + pi_w(+1)).apply(np.log) / (1 + r(+1))
             - (1 + pi_w).apply(np.log))
    return nkwpc, pi_w



# ---------------------------------------------------------------------------
# Government Block
# Fiscal regimes (choose in main.py):
#   DEBT-FINANCED:  b, Tr exogenous; B adjusts
#   TAX-FINANCED:   B fixed; tau adjusts
# b * U + T * V + (1+r(-1)) * B(-1) = tau * w * L + B

@simple
def fiscal(r, w, L, tau, b, Tr, B, U, V):
    LaborTax  = tau * w * L
    BenefCost = b * U + Tr * V
    deficit   = (1 + r(-1)) * B(-1) + BenefCost - LaborTax - B
    return LaborTax, BenefCost, deficit


# Monetary Policy
# Taylor rule:   i = rstar(-1) + phi * pi(-1)  +  Fisher Equation
@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r



# ---------------------------------------------------------------------------
# Market Clearing
@simple
def mkt_clearing(A, B, C, Y, L, U, V):
    asset_mkt = A - B
    labor_mkt = (1 - U - V) - L
    goods_mkt = Y - C
    return asset_mkt, labor_mkt, goods_mkt

