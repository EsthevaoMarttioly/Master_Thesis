#### Thesis - Conditional Cash Transfers: A HANK Approach ####
## Author:  Esthevao Marttioly  |  EESP-FGV  |  2026
## Advisor: Bernardo Guimarães
#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# This program solves an one-asset HANK model with endogenous informality.
# ---------------------------------------------------------------------------
#=
# Write this in the terminal to install packages
# pip install -r requirements.txt

# ---- Packages -------------------------------------------------------------
import random
import numpy as np
from sequence_jacobian import create_model

random.seed(20260415)


# Import parameters
from code.p1_household import hh, solve_ss, solve_dyn
from code.p2_other_blocks import *
from code.p3_parameters import *
from code.p4_results import *


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh, firm_formal, firm_informal, nkpc_ss,
                        union_ss, monetary, fiscal, mkt_clearing])

cal = solve_ss(hank_ss, {**calibration, 'nA': 50},
               unknowns, targets, shares=True, verbose=True)
calibration.update(pi_F = cal['pi_F'], pi_I = cal['pi_I'])

ss = solve_ss(hank_ss, calibration, unknowns, targets, verbose=True)
unknowns = {k: float(ss[k]) for k in unknowns}
calibration.update(unknowns)


# Steady State Diagnostics
print_ss_summary(ss)
tex_macros(ss, calibration, savepath='output/tables/macros.tex')
transition_table(ss, savepath='output/tables/transition_table.tex')


# No-BF Counterfactuals
ss_nobf = solve_ss(hank_ss, {**calibration, 'Tr': 0.0, 'y_bar': 0},
                   unknowns, targets, verbose=True)


# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh, firm_formal, firm_informal,
                     phillips_curve, monetary, fiscal, mkt_clearing])

dyn      = hank.steady_state(ss)
dyn_nobf = hank.steady_state(ss_nobf)

# Verify Dyn is also a valid Steady State
for k in dyn.keys():
    assert np.all(np.isclose(dyn[k], ss[k], atol=1e-6)), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")



# ---------------------------------------------------------------------------
# Equilibrium Jacobians
T = 100
unknowns_dyn  = ['B', 'Y', 'pi', 'w', 'tau']
targets_dyn   = ['debt_rule', 'goods_mkt', 'nkpc', 'wage_nkpc', 'gov_budget']
variables     = ['B', 'C', 'Y', 'L', 'I', 'U', 'BF', 'pi', 'w', 'r', 'i', 'tau']


# IRFs
dTr0, rho_Tr, k = 0.01, 0.40, 4             # Tr: AR(1), rho = 0.4, size = 1%
di0, rho_i      = -0.0025, 0.60             # i:  25bps expansionist, rho = 0.6
dTr = dTr0 * rho_Tr ** np.arange(T)
di  = di0  * rho_i  ** np.arange(T)
dTr_ant = np.concatenate([np.zeros(k), dTr[:-k]])  # Antecipated Shock
di_ant  = np.concatenate([np.zeros(k), di[:-k]])
dTr_perm = dTr0 * np.ones(T)                       # Permanent Shock


## Build IRFs
def build_irfs(shock, dZ, unk=unknowns_dyn, targ=targets_dyn, var=variables, verbose=False):
    run = lambda mv: solve_dyn(hank, dyn, shock, dZ, unk, targ,
                               calibration, var, moving=mv, verbose=verbose)
    insu, full = run(False), run(True)
    comp = {v: full[v] - insu[v] for v in var}
    return dict(insu=insu, full=full, comp=comp)

G_hh     = hh.jacobian(dyn, inputs=['Tr', 'r'], T=T)
irf_tax  = build_irfs('Tr', dTr, unknowns_dyn[1:], targets_dyn[1:], variables[1:])
irf_debt = build_irfs('Tr', dTr)
irf_ant  = solve_dyn(hank, dyn, 'Tr', dTr_ant, unknowns_dyn, targets_dyn, calibration, variables)


irfm_bf   = build_irfs('rstar', di)
irfm_nobf = solve_dyn(hank, dyn_nobf, 'rstar', di, unknowns_dyn, targets_dyn,
                      {**calibration, 'Tr': 0.0}, variables)

irf_pe  = {v: G_hh[v]['Tr'] @ dTr for v in variables if v in G_hh.outputs}
irfm_pe = {v: G_hh[v]['r']  @ di  for v in variables if v in G_hh.outputs}


# ---------------------------------------------------------------------------
# Steady-State - Descriptive Statistics
compare_bf_ss(ss, ss_nobf, savepath='output/tables/ss_comparison.tex')
plot_descriptives(ss, ss_nobf, calibration, savepath='output/figures/bf_descript.png')


# Steady State - Distribution and Policy Functions
plot_consumption_policy(ss, calibration, savepath='output/figures/consump_policy.png')
plot_wealth_distribution(ss, calibration, savepath='output/figures/wealth_distribution.png')


# plot_bf_sweep(lambda cal: solve_ss(hank_ss, cal, unknowns, targets, verbose=True),
#               calibration, ss, ss_nobf, savepath='output/figures/bf_sweep.png')


# ---------------------------------------------------------------------------
# Dynamics - Partial Jacobians
plot_impc(G_hh, savepath='output/figures/impc.png')


# Dynamics - General Jacobians
plot_irf_financing(irf_tax['full'], irf_debt['full'],
                   savepath='output/figures/irf_financing.png')


plot_irf_decomposition(irf_debt['insu'], irf_debt['full'], irf_pe,
                       savepath='output/figures/irf_decomposition.png')


plot_irf_decomposition(irfm_bf['insu'], irfm_bf['full'], irfm_pe,
                       savepath='output/figures/irfm_decomposition.png')


plot_irf({'With BF (total)': irfm_bf['full'], 'Without BF': irfm_nobf,
          'With BF (insurance)': irfm_bf['insu']},
          title='Monetary Policy Shock (i)', savepath='output/figures/irfm_bf_vs_nobf.png')


plot_irf({'Instant Shock': irf_debt['full'], 'Antecipated Shock': irf_ant},
         savepath='output/figures/irf_antecipated.png')


# Dynamics - Cumulative Response
cumulative_response_table(irf_debt['insu'], irf_debt['full'],
                          savepath='output/tables/cumulative_response.tex')


cumulative_response_table(irfm_bf['insu'], irfm_bf['full'], shock='i',
                          savepath='output/tables/monetary_cumulative.tex',
                          label='tab:monetary_cumulative')


rr()
from code.p4_results import *

