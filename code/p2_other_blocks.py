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
def firm_formal(L, Z, w, pi, tau_l, mu, kappa):
    # Formal Sector: Monopolistic Competition with constant Markup.
    Y   = Z * L
    adj = mu / (mu-1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    Div = (1 - tau_l) * (Y - w*L) - adj
    return Y, Div, adj

@simple
def firm_informal(w, N_I):
    # Informal Sector: Perfectly Competitive.
    Y_I = w * N_I
    return Y_I

@simple
def wages(w, N_F, N_I):
    wage_bill = w * (N_F + N_I)
    return wage_bill


# 2. SS Phillips Curve
@simple
def nkpc_ss(w, mu, tau):
    Z = mu * w
    tau_ss = tau
    nkpc = 0.0
    return Z, tau_ss, nkpc


# 3. SS Union's Wage Setting
@simple
def union_ss(w, h_F, C_GHH, L, F, tau_l, mu_w, psi, varphi, eis):
    wage_nkpc = psi * h_F ** (1/varphi) * C_GHH**(1/eis) - (1 - tau_l) * w * L / (h_F * F * mu_w)
    return wage_nkpc


# 4. SS Calibration
@simple
def calibrate_ss(Y, Y_I, N_F, N_I, F, w, L, C_GHH, BF, BF_w, B_gdp, B,
                 r, tau_l, mu_w, eis, h_F, psi, varphi):
    # Invert Market Clearing in Closed Form. Ratios are per Wage Bill.
    L_hat   = N_F                            # Labor Market
    mrp     = (1 - tau_l) * w * L / (F * mu_w * C_GHH ** (1/eis))   # Union FOC, per member
    psi_hat = mrp / h_F ** (1 + 1/varphi)    # Calibration: hours normalized, psi backed out
    h_F_hat = (mrp / psi) ** (varphi/(1+varphi))   # Counterfactual: psi fixed, hours adjust
    Tr_hat  = BF_w * w * (N_F + N_I) / BF    # BF Payment / Wage Bill
    B_hat   = B_gdp * (Y + Y_I)              # Debt / GDP
    tau_hat = tau_l * Y - r * B - Tr_hat * BF   # on the B held: = B_hat once calibrated
    return L_hat, psi_hat, h_F_hat, tau_hat, Tr_hat, B_hat


# 5. Dynamic Phillips Curves
@simple
def phillips_curve(w, r, pi, h_F, Z, Y, L, F, C_GHH, tau_l, mu, mu_w,
                   kappa, kappa_w, eis, psi, varphi, beta_high, dbeta, omega_I):
    beta_avg = beta_high - dbeta * omega_I

    # Price Phillips Curve
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    
    # Wage Phillips Curve
    pi_w = (1 + pi) * w / w(-1) - 1
    wage_nkpc = (kappa_w * (psi * h_F ** (1/varphi) * C_GHH**(1/eis)\
                             - (1 - tau_l) * w * L / h_F / F / mu_w)
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

