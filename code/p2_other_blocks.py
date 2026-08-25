#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the firms, government and monetary policy blocks of the model,
# as well as the market clearing conditions.
#
# SS-DAG   (for calibration)   +   Dynamics DAG   (for IRFs and Jacobians)
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import numpy as np
from sequence_jacobian import simple


# ---------------------------------------------------------------------------
# Firm Block:
# 1. Production
@simple
def firm_formal(Y, Z, w, pi, tau_l, mu, kappa):
    # Formal Sector: Monopolistic Competition with constant Markup.
    L   = Y / Z      # Y = Z * L
    adj = mu / (mu-1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    Div = (1 - tau_l) * (Y - w*L) - adj
    return L, Div, adj

@simple
def firm_informal(w, N_I):
    # Informal Sector: Perfectly Competitive.
    Y_I = w * N_I
    return Y_I


# 2. SS Phillips Curve
@simple
def nkpc_ss(mu, Z, tau):
    w = Z / mu
    nkpc = w / Z - 1 / mu
    tau_ss = tau
    return w, nkpc, tau_ss


# 3. SS Union's Wage Setting
@simple
def union_ss(w, h_F, C_GHH, L, tau_l, mu_w, psi, varphi, eis):
    wage_nkpc = psi * h_F ** (1/varphi) * C_GHH**(1/eis) - (1 - tau_l) * w * L / mu_w
    return wage_nkpc


# 4. SS Calibration, by Direct Inversion
@simple
def calibrate_ss(Y, N_F, w, L, C_GHH, BF, r, B, Tr, tau_l, mu_w, eis, h_F, varphi):
    # Invert Market Clearing in Closed Form.
    Z_hat   = Y / N_F
    psi_hat = (1 - tau_l) * w * L / mu_w / (h_F ** (1/varphi) * C_GHH ** (1/eis))
    tau_hat = tau_l * Y - r * B - Tr * BF
    return Z_hat, psi_hat, tau_hat


# 5. Dynamic Phillips Curves
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
@simple
def fiscal(r, tau_l, Tr, BF, Y, B, tau, tau_ss, B_ss, phi_B):
    BF_Total    = Tr * BF
    tax_revenue = tau_l * Y
    debt_rule   = tau - tau_ss + phi_B * (B(-1) - B_ss)
    gov_budget  = (1 + r) * B(-1) - B + tau + BF_Total - tax_revenue
    return tax_revenue, gov_budget, debt_rule

# Monetary Policy
@simple
def monetary(pi, rstar, phi):
    i = rstar + phi * pi
    r = (1 + i(-1)) / (1 + pi) - 1
    return i, r



# ---------------------------------------------------------------------------
# Market Clearing
@simple
def mkt_clearing(A, B, C, Y, Y_I, L, N_F, adj):
    asset_mkt = A - B
    labor_mkt = N_F - L
    goods_mkt = Y + Y_I - C - adj
    return asset_mkt, labor_mkt, goods_mkt

