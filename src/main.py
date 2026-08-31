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
from src.p1_household import hh
from src.p2_other_blocks import *
from src.p5_calibration import *
from src.p6_solve import *
from src.p7_results import *


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh, firm_formal, firm_informal, nkpc_ss, union_ss,
                        monetary, fiscal, mkt_clearing, calibrate_ss])

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


# No-BF Counterfactuals
ss_nobf = solve_ss(hank_ss, {**calibration, 'Tr': 0.0, 'y_bar': 0}, verbose=True)

compare_bf_ss(ss, ss_nobf, savepath='output/tables/ss_comparison.tex')
plot_descriptives(ss, ss_nobf, calibration, savepath='output/figures/bf_descript.png')

plot_bf_sweep(lambda cal: solve_ss(hank_ss, cal), calibration,
              ss, ss_nobf, savepath='output/figures/bf_sweep.png')



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
dTr     = ar1( 0.01,   0.40, T)             # Tr: AR(1), rho = 0.4, size = 1%
di      = ar1(-0.0025, 0.60, T)             # i:  25bps expansionist, rho = 0.6
dTr_ant = ar1( 0.01,   0.40, T, delay=4)    # Antecipated Shock


## Build IRFs
build_irfs = irf_builder(hank, dyn, calibration, unknowns_dyn, targets_dyn, variables)
build_nobf = irf_builder(hank, dyn_nobf, {**calibration, 'Tr': 0.0},
                         unknowns_dyn, targets_dyn, variables)

G_hh      = hh.jacobian(dyn, inputs=['Tr', 'r'], T=T)
irf_tax   = build_irfs('Tr', dTr, unknowns_dyn[1:], targets_dyn[1:], variables[1:])
irf_debt  = build_irfs('Tr', dTr)
irf_ant   = build_irfs('Tr', dTr_ant, split=False)
irfm_bf   = build_irfs('rstar', di)
irfm_nobf = build_nobf('rstar', di, split=False)

irf_pe    = irf_partial(G_hh, 'Tr', dTr, variables)
irfm_pe   = irf_partial(G_hh, 'r',  di,  variables)



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
from src.p7_results import *

