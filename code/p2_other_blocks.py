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
def firm_formal(L, Z, w, pi, tau_d, mu, kappa):
    # Formal Sector: Monopolistic Competition with constant Markup.
    Y   = Z * L
    adj = mu / (mu-1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    Div = (1 - tau_d) * (Y - w*L) - adj       # Profits taxed at IRPJ + CSLL
    return Y, Div, adj

@simple
def firm_informal(w, N_I):
    # Informal Sector: Perfectly Competitive.
    Y_I = w * N_I
    return Y_I


# 2. Phillips Curves
@simple
def phillips_curve(w, r, pi, h_F, Z, Y, F, UC_F, BETA_F,
                   tau_l, mu, mu_w, kappa, kappa_w, psi, varphi):
    # The union discounts at its own members' beta
    beta_avg = BETA_F / UC_F

    # Price Phillips Curve
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    
    # Wage Phillips Curve
    pi_w = (1 + pi) * w / w(-1) - 1
    mrs  = (1 - tau_l) * w * UC_F / (F * mu_w)
    wage_nkpc = (kappa_w * (psi * h_F ** (1/varphi) - mrs)
                 + beta_avg * (1 + pi_w(+1)).apply(np.log)
                 - (1 + pi_w).apply(np.log))
    return nkpc, wage_nkpc


# 3. Stock Price
@simple
def arbitrage(Div, p_e, r):
    equity = Div(+1) + p_e(+1) - p_e * (1 + r(+1))
    return equity

@simple
def finance(Div, p_e, r, B):
    # Return on the portfolio: = r, except at t = 0
    ra = (Div + p_e + (1 + r) * B(-1)) / (p_e(-1) + B(-1)) - 1
    return ra


# 4. SS Calibration
@simple
def nkpc_ss(w, mu, tau):
    Z = mu * w
    tau_ss = tau
    nkpc = 0.0
    return Z, tau_ss, nkpc

@simple
def union_ss(w, h_F, UC_F, F, tau_l, mu_w, psi, varphi):
    mrs  = (1 - tau_l) * w * UC_F / (F * mu_w)
    wage_nkpc = psi * h_F ** (1/varphi) - mrs
    return wage_nkpc, mrs

@simple
def equity_ss(Div, r):
    ra  = r
    p_e = Div / r
    equity = 0.0
    return p_e, equity, ra

@simple
def calibrate_ss(Y, Y_I, N_F, N_I, w, mrs, BF, BF_w, B_gdp, B,
                 r, tau_l, tau_d, h_F, psi, varphi):
    # Invert Market Clearing in Closed Form. Ratios are per Wage Bill.
    L_hat   = N_F                               # Labor Market
    psi_hat = mrs / h_F ** (1/varphi)           # Calibration:    h_F = 1 (supported by psi)
    h_F_hat = (mrs / psi) ** varphi             # Counterfactual: psi fixed, hours adjust
    Tr_hat  = BF_w * w * (N_F + N_I) / np.maximum(BF, 1e-12)   # BF Payment / Wage Bill
    B_hat   = B_gdp * (Y + Y_I)                 # Debt / GDP
    tax     = tau_l * w * N_F + tau_d * (Y - w * N_F)          # Labor and Profit Taxes
    tau_hat = tax - r * B - Tr_hat * BF         # on the B held: = B_hat once calibrated
    return L_hat, psi_hat, h_F_hat, tau_hat, Tr_hat, B_hat



# ---------------------------------------------------------------------------
# Government Block
@simple
def fiscal(r, tau_l, tau_d, Tr, BF, Y, w, L, B, tau, tau_ss, B_ss, phi_B):
    BF_Total    = Tr * BF
    tax_revenue = tau_l * w * L + tau_d * (Y - w * L)
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
def mkt_clearing(A, B, p_e, C, Y, Y_I, L, N_F, adj):
    asset_mkt = A - B - p_e
    labor_mkt = N_F - L
    goods_mkt = Y + Y_I - C - adj
    return asset_mkt, labor_mkt, goods_mkt

