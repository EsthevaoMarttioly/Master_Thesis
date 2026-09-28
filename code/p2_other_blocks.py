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
def firm_informal(Z_I, N_I, w_I, alpha_I):
    # Informal Sector: Decreasing Returns
    Y_I = w_I * N_I
    informal = w_I - Z_I * N_I ** (alpha_I - 1)
    return Y_I, informal


# 2. Phillips Curves
@simple
def phillips_curve(w, w_I, r, pi, h_F, h_I, Z, Y, F, I, UC_F, UC_I, BETA_F, BETA_I,
                   tau_l, mu, mu_w, kappa, kappa_w, kappa_wI, alpha_I, psi, varphi):
    # Each union discounts at its own members' beta
    beta_F, beta_I = BETA_F / UC_F, BETA_I / UC_I

    # Price Phillips Curve
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    
    # Wage Phillips Curves
    pi_w   = (1 + pi) * w / w(-1) - 1
    pi_wI  = (1 + pi) * w_I / w_I(-1) - 1
    mrs    = (1 - tau_l) * w * UC_F / (F * mu_w)
    mrs_I  = alpha_I * w_I * UC_I / (I * mu_w)

    wage_nkpc = (kappa_w * (psi * h_F ** (1/varphi) - mrs)
                 + beta_F * (1 + pi_w(+1)).apply(np.log)
                 - (1 + pi_w).apply(np.log))

    wage_nkpc_I = (kappa_wI * (psi * h_I ** (1/varphi) - mrs_I)
                   + beta_I * (1 + pi_wI(+1)).apply(np.log)
                   - (1 + pi_wI).apply(np.log))
    return nkpc, wage_nkpc, wage_nkpc_I


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
def nkpc_ss(w, mu, tau_l):
    Z = mu * w
    tau_ss = tau_l
    nkpc = 0.0
    return Z, tau_ss, nkpc

@simple
def union_ss(w, w_I, h_F, h_I, UC_F, UC_I, F, I, tau_l, mu_w, alpha_I, psi, varphi):
    mrs   = (1 - tau_l) * w * UC_F / (F * mu_w)
    mrs_I = alpha_I * w_I * UC_I / (I * mu_w)
    wage_nkpc   = psi * h_F ** (1/varphi) - mrs
    wage_nkpc_I = psi * h_I ** (1/varphi) - mrs_I
    return wage_nkpc, wage_nkpc_I, mrs, mrs_I

@simple
def equity_ss(Div, r):
    ra  = r
    p_e = Div / r
    equity = 0.0
    return p_e, equity, ra

@simple
def calibrate_ss(Y, Y_I, N_F, N_I, w, w_I, Z_I, mrs, UC_I, I, BF, BF_w, B_gdp, B, G,
                 r, tau_l, tau_d, mu_w, alpha_I, h_F, psi, varphi):
    # Invert Market Clearing in Closed Form. Ratios are per Wage Bill.
    L_hat   = N_F                               # Labor Market
    psi_hat = mrs / h_F ** (1/varphi)           # Calibration:    h_F = 1 (supported by psi)
    h_F_hat = (mrs / psi) ** varphi             # Counterfactual: psi fixed, hours adjust
    Z_I_hat = w * N_I ** (1 - alpha_I)          # Calibration:    w_I = w (supported by Z_I)
    w_I_hat = Z_I * N_I ** (alpha_I - 1)        # Counterfactual: Z_I fixed, w_I adjusts
    h_I_hat = (alpha_I * w_I * UC_I / (I * mu_w * psi)) ** varphi
    Tr_hat  = BF_w * w * (N_F + N_I) / np.maximum(BF, 1e-12)   # BF Payment / Wage Bill
    B_hat   = B_gdp * (Y + Y_I)                                # Debt / GDP
    tax     = tau_l * w * N_F + tau_d * (Y - w * N_F)          # Labor and Profit Taxes
    G_hat   = tax - r * B - Tr_hat * BF         # on the B held: = B_hat once calibrated
    tau_l_hat = (G + r * B + Tr_hat * BF - tau_d * (Y - w * N_F)) / (w * N_F)
    return L_hat, psi_hat, h_F_hat, h_I_hat, Z_I_hat, w_I_hat, G_hat, Tr_hat, B_hat, tau_l_hat



# ---------------------------------------------------------------------------
# Government Block
@simple
def fiscal(r, tau_l, tau_d, Tr, BF, G, Y, w, L, B, tau_ss, B_ss, phi_B):
    BF_Total    = Tr * BF
    tax_revenue = tau_l * w * L + tau_d * (Y - w * L)
    debt_rule   = tau_l - tau_ss - phi_B * (B(-1) - B_ss) / (w * L)
    gov_budget  = G + (1 + r) * B(-1) - B + BF_Total - tax_revenue
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
def mkt_clearing(A, B, p_e, C, G, Y, Y_I, U, y_u, L, N_F, adj):
    asset_mkt = A - B - p_e
    labor_mkt = N_F - L
    goods_mkt = Y + Y_I + y_u * U - C - G - adj
    return asset_mkt, labor_mkt, goods_mkt

