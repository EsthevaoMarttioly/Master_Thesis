#=
#---------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid and transition matrices for income, and the labor income function.
#---------------------------------------------------------------------------
#=

# Import Packages
import numpy as np
from sequence_jacobian import simple


#---------------------------------------------------------------------------
# Firm Block:
# Production: Y = Z * L  =>  L = Y / Z
@simple
def firm(Y, Z):
    L = Y / Z
    return L


# Phillips Curves:
# log(1+pi) = kappa * (w/Z - 1/mu) + Y(+1)/Y * log(1+pi(+1)) / (1+r(+1))
@simple
def price_nkpc(pi, w, Z, Y, r, kappa, mu):
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    return nkpc


# log(1+pi^w) = kappa_w * (w/Z - 1/mu_w) + log(1+pi+w(+1)) / (1+r(+1))
@simple
def wage_nkpc(pi, w, Z, r, kappa_w, mu_w):
    pi_w  = (1 + pi) * w / w(-1) - 1
    nkwpc = (kappa_w * (w / Z - 1 / mu_w)
             + (1 + pi_w(+1)).apply(np.log) / (1 + r(+1))
             - (1 + pi_w).apply(np.log))
    return nkwpc, pi_w


@simple
def dividends(Y, w, L):
    div = Y - w * L
    return div



#---------------------------------------------------------------------------
# Government Block
# Budget constraint:   b_t * U_t + T_t * V_t + (1+r_{t-1})*B_{t-1} = tau*w_t*L_t + B_t
@simple
def fiscal(r, w, L, U, V, tau, b, Tr, B):
    LaborTax = tau * w * L
    BenefCost = b * U + Tr * V
    deficit   = (1 + r(-1)) * B(-1) + BenefCost - LaborTax - B
    return LaborTax, BenefCost, deficit


# Monetary Policy
# Taylor rule:   i = rstar(-1) + phi * pi(-1)  +  Fisher Equation
@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r



#---------------------------------------------------------------------------
# Market Clearing
@simple
def mkt_clearing(A, B, C, Y, L, U, V):
    asset_mkt = A - B
    labor_mkt = (1 - U - V) - L
    goods_mkt = Y - C
    return asset_mkt, labor_mkt, goods_mkt

