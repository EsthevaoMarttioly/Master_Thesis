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
    Div = (1 - tau_l) * (Y - w*L)
    return L, Div


@simple
def firm_informal(w_I, N_I):
    # Informal sector: perfectly competitive.
    Y_I = w_I * N_I
    return Y_I



# 2. SS Phillips Curve
@simple
def nkpc_ss(mu, Z):
    w = Z / mu
    return w


@simple
def informal_wage(w, xi):
    w_I = xi * w
    return w_I



# 3. SS Union's Wage Setting
@simple
def union_ss(w, h_F, C_GHH, L, tau_l, mu_w, psi, varphi, eis):
    wage_nkpc = psi * h_F ** (1/varphi) * C_GHH**(1/eis) - (1 - tau_l) * w * L / mu_w
    return wage_nkpc



# 4. Dynamic Phillips Curves
@simple
def phillips_curve(w, r, pi, h_F, Z, Y, L, C_GHH, tau_l, mu, mu_w,
                   kappa, kappa_w, eis, psi, varphi, beta_high, dbeta, omega_I):
    beta_avg = beta_high - dbeta * omega_I

    # Price Phillips Curve
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    
    # Wage Phillips Curve
    pi_w = (1 + pi) * w / w(-1) - 1
    wage_nkpc = (kappa_w * (psi * h_F ** (1/varphi) * C_GHH**(1/eis)\
                             - (1 - tau_l) * w * L / mu_w)
                 + beta_avg * (1 + pi_w(+1)).apply(np.log)
                 - (1 + pi_w).apply(np.log))
    return nkpc, wage_nkpc



# ---------------------------------------------------------------------------
# Government Block
# Fiscal regimes (choose in main.py):
#   DEBT-FINANCED:  b, Tr exogenous; B adjusts
#   TAX-FINANCED:   B fixed; tau adjusts
# G + Tr * BF + (1+r) * B(-1) = tau * w * F + B

@simple
def fiscal(r, tau_l, Tr, BF, Y, B, G):
    tax_revenue = tau_l * Y
    BF_Total    = Tr * BF
    gov_budget  = (1 + r) * B(-1) - B + G + BF_Total - tax_revenue
    return tax_revenue, gov_budget


# Monetary Policy
# Taylor rule:   i = rstar + phi * pi  +  Fisher Equation
@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar + phi * pi(-1)) / (1 + pi) - 1
    return r



# ---------------------------------------------------------------------------
# Market Clearing
@simple
def mkt_clearing(A, B, C_GHH, Y, Y_I, G, L, N_F, varphi):
    C = C_GHH + varphi/(1+varphi) * Y_I
    asset_mkt = A - B
    labor_mkt = N_F - L
    goods_mkt = Y + Y_I - C - G
    return C, asset_mkt, labor_mkt, goods_mkt

