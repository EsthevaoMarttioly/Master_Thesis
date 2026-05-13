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
def firm(Y, Z, w, tau):
    L   = Y / Z       # Y = Z * L  =>  L = Y / Z
    Div = (1-tau) * (Y - w*L)
    return L, Div


# 2. SS Phillips Curve
#    wedge = vscale * L^(1/frisch) - (1-tau) * w * C^(-1/eis)
#    pi = kappa * [wedge] + beta_avg * pi(+1)
@simple
def nkpc_ss(Z, mu):
    w = Z / mu        # P = mu * W / Z  =>  w = Z / mu
    return w


# 3. Dynamic Phillips Curves
@simple
def phillips_curve(pi, w, L, C, kappa, frisch, eis, vscale, tau, beta_high, omega_I, dbeta):
    """Single wage-price Phillips curve
    Returns nkpc residual = 0 in equilibrium"""
    beta_avg = beta_high - omega_I * dbeta
    wedge = vscale * L**(1/frisch) - (1 - tau) * w * C**(-1/eis)
    nkpc  = kappa * wedge + beta_avg * pi(+1) - pi
    return nkpc



# ---------------------------------------------------------------------------
# Government Block
# Fiscal regimes (choose in main.py):
#   DEBT-FINANCED:  b, Tr exogenous; B adjusts
#   TAX-FINANCED:   B fixed; tau adjusts
# b * U + T * V + (1+r(-1)) * B(-1) = tau * w * L + B

@simple
def fiscal(r, Y, tau, b, Tr, B, U, V):
    TotalTax   = tau * Y
    Transfers  = b * U + Tr * V
    gov_budget = (1 + r(-1)) * B(-1) + Transfers - TotalTax - B
    return TotalTax, Transfers, gov_budget


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

