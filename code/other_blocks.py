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
def firm_formal(Y, Z, w, tau_l):
    # Formal sector: monopolistic competition with constant markup.
    L   = Y / Z       # Y = Z * L  =>  L = Y / Z
    Div = Y - w*L
    return L, Div


@simple
def firm_informal(xi, w, Y_I):
    # Informal sector: perfectly competitive.
    w_I = xi * w
    L_I = Y_I / w_I
    return w_I, L_I


# 2. SS Phillips Curve
@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w


# 3. Dynamic Phillips Curves
@simple
def phillips_curve(pi, w, Z, Y, r, kappa, mu):
    # Price Phillips Curve
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    return nkpc


@simple
def wage_phillips_curve(pi, w, Z, r, kappa_w, mu, phi_out, outside_opt):
    # Wage Phillips Curve
    pi_w = (1 + pi) * w / w(-1) - 1
    wage_nkpc = (kappa_w * (w / Z - 1 / mu + phi_out * outside_opt)
                 + (1 + pi_w(+1)).apply(np.log) / (1 + r(+1))
                 - (1 + pi_w).apply(np.log))
    return wage_nkpc, pi_w



#----------------------------------------------------------------------------
# Sector Transition
@simple
def sector_flows(w, w_I, tau_l, Tr, lambda_s, eta_s, p_fi, p_if):
    """Endogenous sector transition rates.
    Switching rates (logistic):
        sigma   = logistic(eta_s * outside_opt)  \in  (0,1)
        p_fi    = lambda_s * sigma
        p_if    = lambda_s * (1 - sigma)"""

    outside_opt = (w_I + Tr) / ((1 - tau_l) * w) - 1

    sigma = 1 / (1 + np.exp(-eta_s * outside_opt))
    sector_fi = p_fi - lambda_s * sigma
    sector_if = p_if - lambda_s * (1 - sigma)

    return sector_fi, sector_if, outside_opt



# ---------------------------------------------------------------------------
# Government Block
# Fiscal regimes (choose in main.py):
#   DEBT-FINANCED:  b, Tr exogenous; B adjusts
#   TAX-FINANCED:   B fixed; tau adjusts
# G + Tr * BF + (1+r(-1)) * B(-1) = tau * w * F + B

@simple
def fiscal(r, w, tau_l, Tr, BF, F, B, G):
    tax_revenue = tau_l * w * F
    BF_Total    = Tr * BF
    gov_budget  = (1 + r(-1)) * B(-1) - B + G + BF_Total - tax_revenue
    return tax_revenue, gov_budget


# Monetary Policy
# Taylor rule:   i = rstar(-1) + phi * pi(-1)  +  Fisher Equation
@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r



# ---------------------------------------------------------------------------
# Market Clearing
@simple
def mkt_clearing(A, B, C, Y, Y_I, G, L, F, N_I, L_I, varphi):
    asset_mkt = A - B
    formal_labor_mkt = F - L
    informal_labor_mkt = N_I - L_I
    goods_mkt = Y + 1/(1+varphi) * Y_I - C - G
    return asset_mkt, formal_labor_mkt, informal_labor_mkt, goods_mkt

