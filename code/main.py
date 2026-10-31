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
import numpy as np
from sequence_jacobian import create_model

# Import parameters
from code.p1_household import hh, hh_ss
from code.p2_other_blocks import *
from code.p5_calibration import *
from code.p6_solve import *
from code.p7_smm import load_res, report
from code.p8_results import *
from code.p9_robustness import *


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh_ss, firm_formal, firm_informal, nkpc_ss, union_ss,
                        equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss])

ss = solve_ss(hank_ss, calibration, flows, damp=0.5, verbose=True)
calibration.update({k: ss[k] for k in update + ['Pi', 'Qb']})


# Steady State Diagnostics
print_ss_summary(ss)
tex_macros(ss, calibration, savepath='output/tables/macros.tex')
transition_table(ss, savepath='output/tables/transition_table.tex')
report(load_res(), savepath='output/tables/moments.tex')


# Distribution and Policy Functions
plot_consumption_policy(ss, calibration, savepath='output/figures/consump_policy.png')
plot_wealth_distribution(ss, savepath='output/figures/wealth_distribution.png')
plot_income_distribution(ss, savepath='output/figures/income_distribution.png')
plot_descriptives(ss, savepath='output/figures/bf_descript.png')
plot_bf_targeting(ss, savepath='output/figures/bf_distribution.png')



# ---------------------------------------------------------------------------
# Counterfactuals
solve_cf = lambda cal, **kw: solve_ss(hank_ss, cal, verbose=False, **kw)
cfs      = counterfactuals(solve_cf)
(ss_nobf, calib_nobf), (ss_perf, calib_perf) = cfs.values()

compare_ss({'Baseline': ss, **{n: s for n, (s, _) in cfs.items()}},
           savepath='output/tables/ss_counterfactuals.tex')
compare_bf_ss(ss, ss_nobf, savepath='output/tables/ss_comparison.tex')
plot_bf_comparison(ss, ss_nobf, savepath='output/figures/bf_comparison.png')



# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh.remap({'r': 'ra'}), firm_formal, firm_informal,
                     phillips_curve, arbitrage, finance, monetary, fiscal, mkt_clearing])

dyn      = hank.steady_state(ss)
dyn_nobf = hank.steady_state(ss_nobf)
dyn_perf = hank.steady_state(ss_perf)

# Verify Dyn is also a valid Steady State
for k in dyn.keys():
    assert np.all(np.isclose(dyn[k], ss[k], atol=1e-5)), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")


# Shocks
T   = 150                               # Solve horizon
wdw = 40                                # Convergence Window
Td  = 80                                # Reported Horizon
dTr = ar1( 0.10, 0.40, T)               # Tr: +10%, then AR(1) with rho = 0.4
di  = hike(0.50, 0.50, T, 1)            # i:  Copom tightening, +50bps, then AR(1)


## Market Clearing Targets
unknowns_dyn = ['B', 'L', 'h_F', 'h_I', 'pi', 'w', 'w_I', 'tau_c', 'p_e']
targets_dyn  = ['debt_rule', 'asset_mkt', 'labor_mkt', 'informal',
                'nkpc', 'wage_nkpc', 'wage_nkpc_I', 'gov_budget', 'equity']
variables    = ['B', 'tau_c', 'C', 'Y', 'Y_I', 'L', 'N_I', 'I', 'U', 'BF', 'pi', 'w', 'w_I', 'r', 'i']


## IRFs: Fiscal (Tr) and Monetary (i) Shocks
build_irfs = irf_builder(hank, dyn, calibration, unknowns_dyn, targets_dyn, variables, wdw, Td)
build_nobf = irf_builder(hank, dyn_nobf, calib_nobf, unknowns_dyn, targets_dyn, variables, wdw, Td)
build_perf = irf_builder(hank, dyn_perf, calib_perf, unknowns_dyn, targets_dyn, variables, wdw, Td)

G_hh      = hh.jacobian(dyn, inputs=['Tr', 'r', 'tau_c', 'w'], T=T)
irf_tr    = build_irfs('Tr', dTr)
irfm_bf   = build_irfs('rstar', di)
irfm_nobf = build_nobf('rstar', di, split=False)
irfm_perf = build_perf('rstar', di, split=False)
irf_pe    = irf_partial(G_hh, 'Tr', dTr, variables)
irfm_pe   = irf_partial(G_hh, 'r',  di,  variables)



# ---------------------------------------------------------------------------
# Dynamics - Graphics
plot_impc(G_hh, savepath='output/figures/impc.png')


plot_irf_decomposition(irf_tr['insu'], irf_tr['full'], irf_pe,
                       savepath='output/figures/irf_decomposition.png')


plot_irf_decomposition(irfm_bf['insu'], irfm_bf['full'], irfm_pe,
                       savepath='output/figures/irfm_decomposition.png')


plot_irf({'With BF (Total)': irfm_bf['full'], 'Without BF': irfm_nobf,
          'With BF (Insurance)': irfm_bf['insu']},
          title='Monetary Policy Shock (i)', savepath='output/figures/irfm_bf_vs_nobf.png')


# Dynamics - Tables (Cumulative Response)
cumulative_response_table(irf_tr['insu'], irf_tr['full'],
                          savepath='output/tables/cumulative_response.tex')


cumulative_response_table(irfm_bf['insu'], irfm_bf['full'], shock='i',
                          savepath='output/tables/monetary_cumulative.tex',
                          label='tab:monetary_cumulative')



# ---------------------------------------------------------------------------
# Robustness Tests
solve = lambda cal, **kw: solve_ss(hank_ss, cal, flows=flows, **kw)

def build(cal):
    c = {**calibration, **cal}
    d = hank.steady_state(solve(c)) if set(cal) & dyn_ss else dyn
    return irf_builder(hank, d, c, unknowns_dyn, targets_dyn, variables, Td)

## Steady State
rob  = robustness(solve, calibration)
rob |= bf_sizes(solve, calibration)
robustness_table(rob, savepath='output/tables/robustness.tex')
bf_size_table(rob, savepath='output/tables/robustness_bf.tex')


## Dynamics
dyn  = dyn_robustness(build, 'Tr', dTr, moving=False)
dyn_rob_table(dyn, savepath='output/tables/dyn_robustness.tex')
plot_dyn_robustness(dyn, savepath='output/figures/dyn_robustness.png')
plot_dyn_robustness(dyn, {'alpha_I': dyn_grid['alpha_I']},
                    savepath='output/figures/dyn_robust_phi.png')





# ---------------------------------------------------------------------------
# Dynamics - Financing and Timing
dTr_prm = ar1(0.10 * 0.05/0.60, 0.95, T)         # Persistent
dTr_ant = {d: ar1(0.10, 0.40, T, delay=d) for d in (1, 4, 8)}

irf_tr_tax  = build_irfs('Tr', dTr, unknowns_dyn[1:], targets_dyn[1:], variables[1:])
irf_tr_prm  = build_irfs('Tr', dTr_prm, split=False)
irf_tr_ant  = {d: build_irfs('Tr', dZ, split=False) for d, dZ in dTr_ant.items()}


# Zero-Distortion: no debt and no tax
unk_G    = ['G' if k == 'tau_c' else k for k in unknowns_dyn if k != 'B']
tgt_G    = [t for t in targets_dyn if t != 'debt_rule']
irf_tr_G = build_irfs('Tr', dTr, unk_G, tgt_G, variables[2:], split=False)


# Dynamics - Financing and Timing
plot_irf_financing(irf_tr_tax['full'], irf_tr['full'], ('Tax-Financed', 'Debt-Financed'),
                   savepath='output/figures/irf_financing.png')


plot_irf_financing(irf_tr_G, irf_tr['full'], ('Zero-Distortion', 'Debt-Financed'),
                   savepath='output/figures/irf_financing_G.png')


plot_irf({'rho = 0.40': irf_tr['full'], 'rho = 0.95 (Highly-Persistent)': irf_tr_prm},
         title='Transfer Shock: Persistence', savepath='output/figures/irf_persistence.png')


plot_irf({'Unanticipated': irf_tr['full'],
          **{f'Announced {d}q ahead': v for d, v in irf_tr_ant.items()}},
         savepath='output/figures/irf_anticipated.png')



# ---------------------------------------------------------------------------
import importlib, code.p6_solve, code.p8_results, code.p9_robustness
importlib.reload(code.p6_solve)
importlib.reload(code.p8_results)
importlib.reload(code.p9_robustness)
globals().update({k: v for k, v in vars(code.p6_solve).items()
                    if not k.startswith('_')})
globals().update({k: v for k, v in vars(code.p8_results).items()
                    if not k.startswith('_')})
globals().update({k: v for k, v in vars(code.p9_robustness).items()
                    if not k.startswith('_')})
from code.p6_solve import *
from code.p8_results import *
from code.p9_robustness import *