#### Thesis - CCTs as a Stabilization Tool: a HANK Approach ####
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
from code.p1_household import hh
from code.p2_other_blocks import *
from code.p5_calibration import *
from code.p6_solve import *
from code.p7_results import *


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh, firm_formal, firm_informal, wages, nkpc_ss,
                        union_ss, monetary, fiscal, mkt_clearing, calibrate_ss])

ss = solve_ss(hank_ss, calibration, flows, verbose=True)
calibration.update({k: float(ss[k]) for k in (*pi_calib, *unknowns, 'tau_ss', 'B_ss')})


# Steady State Diagnostics
print_ss_summary(ss)
tex_macros(ss, calibration, savepath='output/tables/macros.tex')
transition_table(ss, savepath='output/tables/transition_table.tex')


# Distribution and Policy Functions
plot_consumption_policy(ss, calibration, savepath='output/figures/consump_policy.png')
plot_wealth_distribution(ss, savepath='output/figures/wealth_distribution.png')
plot_income_distribution(ss, savepath='output/figures/income_distribution.png')
plot_descriptives(ss, savepath='output/figures/bf_descript.png')


# No-BF Counterfactuals   -   Change with Dynamics
ss_nobf = solve_ss(hank_ss, {**calibration, 'BF_w': 0.0}, verbose=True)

compare_bf_ss(ss, ss_nobf, savepath='output/tables/ss_comparison.tex')
plot_descriptives(ss, ss_nobf, savepath='output/figures/bf_descript.png')

# plot_bf_sweep(lambda cal: solve_ss(hank_ss, cal), calibration,
#               ss, ss_nobf, savepath='output/figures/bf_sweep.png')



# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh, firm_formal, firm_informal, wages,
                     phillips_curve, monetary, fiscal, mkt_clearing])

dyn      = hank.steady_state(ss)
dyn_nobf = hank.steady_state(ss_nobf)

# Verify Dyn is also a valid Steady State
for k in dyn.keys():
    assert np.all(np.isclose(dyn[k], ss[k], atol=1e-5)), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")



# ---------------------------------------------------------------------------
# Dynamics - Shocks
T = 100
dTr     = ar1( 0.01,    0.40, T)             # Tr: AR(1), rho = 0.4, size = 1%
di      = hike(-0.5, 4, 0.85, T)             # i:  Copom cycle, -50bps/quarter for 1y
dTr_ant = ar1( 0.01,    0.40, T, delay=4)    # Antecipated Shock


## Market Clearing Targets
unknowns_dyn = ['B', 'L', 'h_F', 'pi', 'w', 'tau']
targets_dyn  = ['debt_rule', 'asset_mkt', 'labor_mkt', 'nkpc', 'wage_nkpc', 'gov_budget']
variables    = ['B', 'C', 'Y', 'L', 'I', 'U', 'BF', 'pi', 'w', 'r', 'i', 'tau']


## IRFs: Fiscal (Tr) and Monetary (i) Shocks
build_irfs = irf_builder(hank, dyn, calibration, unknowns_dyn, targets_dyn, variables)
build_nobf = irf_builder(hank, dyn_nobf, {**calibration, 'Tr': 0.0},
                         unknowns_dyn, targets_dyn, variables)

G_hh      = hh.jacobian(dyn, inputs=['Tr', 'r'], T=T)
irf_tr    = build_irfs('Tr', dTr)
irfm_bf   = build_irfs('rstar', di)
irfm_nobf = build_nobf('rstar', di, split=False)
irf_pe    = irf_partial(G_hh, 'Tr', dTr, variables)
irfm_pe   = irf_partial(G_hh, 'r',  di,  variables)


# Permanent Removal: start at the BF distribution, travel to the No-BF Steady State
irf_kill  = permanent(hank, dyn_nobf, dyn, unknowns_dyn, targets_dyn, calibration,
                      variables, T=300, moving=False, tol=1e-5, verbose=True)



# ---------------------------------------------------------------------------
# Dynamics - Partial Jacobians
plot_impc(G_hh, savepath='output/figures/impc.png')


# Dynamics - Fiscal and Monetary
plot_irf_decomposition(irf_tr['insu'], irf_tr['full'], irf_pe,
                       savepath='output/figures/irf_decomposition.png')


plot_irf_decomposition(irfm_bf['insu'], irfm_bf['full'], irfm_pe,
                       savepath='output/figures/irfm_decomposition.png')


plot_irf({'With BF (Total)': irfm_bf['full'], 'Without BF': irfm_nobf,
          'With BF (Insurance)': irfm_bf['insu']},
          title='Monetary Policy Shock (i)', savepath='output/figures/irfm_bf_vs_nobf.png')


plot_irf({'Permanent Removal': rebase(irf_kill, dyn_nobf, dyn)}, T_plot=120,
         title='Permanent Removal of Bolsa Familia',
         savepath='output/figures/irf_nobf_permanent.png')


# Dynamics - Cumulative Response
cumulative_response_table(irf_tr['insu'], irf_tr['full'],
                          savepath='output/tables/cumulative_response.tex')


cumulative_response_table(irfm_bf['insu'], irfm_bf['full'], shock='i',
                          savepath='output/tables/monetary_cumulative.tex',
                          label='tab:monetary_cumulative')



# ---------------------------------------------------------------------------
# Dynamics - Financing and Timing
irf_tr_tax = build_irfs('Tr', dTr, unknowns_dyn[1:], targets_dyn[1:], variables[1:])
irf_tr_ant = build_irfs('Tr', dTr_ant, split=False)


plot_irf_financing(irf_tr_tax['full'], irf_tr['full'],
                   savepath='output/figures/irf_financing.png')


plot_irf({'Instant Shock': irf_tr['full'], 'Antecipated Shock': irf_tr_ant},
         savepath='output/figures/irf_antecipated.png')


rr(); from code.p7_results import *


