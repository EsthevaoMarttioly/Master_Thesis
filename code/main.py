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

figures = 'output/figures/model/'
tables  = 'output/tables/'


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh_ss, firm_formal, firm_informal, nkpc_ss, union_ss,
                        equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss])

ss = solve_ss(hank_ss, calibration, flows, damp=0.0, verbose=True)
calibration.update({k: ss[k] for k in update + ['Pi', 'Qb']})


# Steady State Diagnostics
print_ss_summary(ss)
tex_macros(ss, calibration, savepath=f'{tables}macros.tex')
transition_table(savepath=f'{tables}transition_table.tex')
report(load_res(), savepath=f'{tables}moments.tex')


# Distribution and Policy Functions
plot_consumption_policy(ss, calibration, savepath=f'{figures}consump_policy.png')
plot_wealth_distribution(ss, savepath=f'{figures}wealth_distribution.png')
plot_income_distribution(ss, savepath=f'{figures}income_distribution.png')
plot_descriptives(ss, savepath=f'{figures}bf_descript.png')
plot_bf_targeting(ss, savepath=f'{figures}bf_distribution.png')



# ---------------------------------------------------------------------------
# Counterfactuals
solve_cf = lambda cal, **kw: solve_ss(hank_ss, cal, verbose=False, **kw)
cfs      = counterfactuals(solve_cf)
(ss_nobf, calib_nobf), (ss_compl, calib_compl) = cfs.values()

compare_ss({'Baseline': ss, **{n: s for n, (s, _) in cfs.items()}},
           savepath=f'{tables}ss_counterfactuals.tex')
compare_bf_ss(ss, ss_nobf, savepath=f'{tables}ss_comparison.tex')
plot_bf_comparison(ss, ss_nobf, savepath=f'{figures}bf_comparison.png')



# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh.remap({'r': 'ra'}), firm_formal, firm_informal,
                     phillips_curve, arbitrage, finance, monetary, fiscal, mkt_clearing])

dyn       = hank.steady_state(ss)
dyn_nobf  = hank.steady_state(ss_nobf)
dyn_compl = hank.steady_state(ss_compl)

# Verify Dyn is also a valid Steady State
for k in dyn.keys():
    assert np.all(np.isclose(dyn[k], ss[k], atol=1e-5)), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")


# Shocks
T   = 100                               # Solve horizon
wdw = 60                                # Convergence Window
dTr = ar1(0.01,       0.90, T)          # Tr: +1%, then AR(1) with rho = 0.9
di  = ar1(0.50 / 400, 0.50, T)          # i:  Tightening, +50bps, then AR(1)
di_ant = {d: ar1(0.50 / 400, 0.50, T, delay=d) for d in (1, 4, 8)}     # Anticipated


## Market Clearing Targets
unk_dyn  = ['B', 'tau_c', 'pi', 'w', 'h_F', 'h_I', 'L', 'p_e']
tgt_dyn  = ['gov_budget', 'debt_rule', 'nkpc', 'wage_nkpc', 'labor_mkt',  'wage_nkpc_I', 'asset_mkt',  'equity']
var      = ['B', 'tau_c', 'C', 'Y', 'Y_I', 'h_I', 'I', 'U', 'BF', 'pi', 'w', 'r']

unk_tax = [k for k in unk_dyn  if k != 'B']
tgt_tax = [k for k in tgt_dyn   if k != 'debt_rule']
var_tax = [k for k in var if k != 'B']


## IRFs: Fiscal (Tr) and Monetary (i) Shocks
G_hh = hh.jacobian(dyn, inputs=['Tr', 'r', 'tau_c', 'w'], T=T)
build_irfs = irf_builder(hank, dyn, calibration, unk_tax, tgt_tax, var_tax, wdw)
build_nobf = irf_builder(hank, dyn_nobf, calib_nobf, unk_tax, tgt_tax, var_tax, wdw)
build_compl = irf_builder(hank, dyn_compl, calib_compl, unk_tax, tgt_tax, var_tax, wdw)


unk_G    = ['G' if k == 'tau_c' else k for k in unk_tax]
var_G    = [k for k in var if k not in ['tau_c', 'B']]
debt_fin = dict(unk=unk_dyn, targ=tgt_dyn, v=var)
debt_G   = dict(unk=unk_G, targ=tgt_tax, v=var_G)

cases = {'irfm_bf'    : (build_irfs,  'rstar', di,  dict()),
         'irfm_nobf'  : (build_nobf,  'rstar', di,  dict(split=False)),
         'irfm_compl' : (build_compl, 'rstar', di,  dict(split=False)),
         'irf_tr'     : (build_irfs,  'Tr',    dTr, dict(debt_fin, gpi=0.6, adapt=False)),
         'irf_tax'    : (build_irfs,  'Tr',    dTr, dict(split=False)),
         'irf_G'      : (build_irfs,  'Tr',    dTr, dict(debt_G, split=False)),
         'irfm_G'     : (build_irfs,  'rstar', di,  dict(debt_G, split=False)),
         'irfm_G_nobf': (build_nobf,  'rstar', di,  dict(debt_G, split=False)),
         **{f'irfm_ant{d}': (build_irfs, 'rstar', dZ, dict(split=False)) for d, dZ in di_ant.items()}}

irfs = solve_cases(cases, path='output/smm/irfs.pkl', verbose=True)

irfm_bf, irfm_nobf, irfm_compl, irf_tr, irf_tax, irf_G, irfm_G, irfm_G_nobf = (irfs[k] for k in list(cases)[:8])
irfm_ant = {d: irfs[f'irfm_ant{d}'] for d in di_ant}



# ---------------------------------------------------------------------------
# Dynamics - Graphics
plot_impc(G_hh, savepath=f'{figures}impc.png')


# Fiscal
plot_irf_decomposition(irf_tr['benef'], irf_tr['full'], T_plot=40,
                       title='Transfer Shock (+1%)', savepath=f'{figures}irf_decomposition.png')


plot_irf_financing({'Debt-Financed': irf_tr['full'], 'Tax-Financed': irf_tax,
                    'Zero-Distortion': irf_G}, savepath=f'{figures}irf_financing.png')


# Monetary
plot_irf_decomposition(irfm_bf['benef'], irfm_bf['full'], savepath=f'{figures}irfm_decomposition.png')


plot_irf({'With BF (Total)': irfm_bf['full'], 'With BF (Benefit)': irfm_bf['benef'],
          'Without BF': irfm_nobf}, savepath=f'{figures}irfm_bf_vs_nobf.png')


plot_irf({'With BF': irfm_bf['full'], 'Total Compliance': irfm_compl,
          'Without BF': irfm_nobf,}, savepath=f'{figures}irfm_compliance.png')


plot_irf_financing({'Tax-Financed': irfm_bf['full'], 'Zero-Distortion': irfm_G,
                    'Without BF': irfm_G_nobf}, savepath=f'{figures}irfm_G.png')


plot_irf({'Unanticipated': irfm_bf['full'],
          **{f'{d}q ahead': v for d, v in irfm_ant.items()}},
          savepath=f'{figures}irfm_anticipated.png')


# Dynamics - Tables (Cumulative Response)
cumulative_response_table(irf_tr['benef'], irf_tr['full'], savepath=f'{tables}cumulative_response.tex')


cumulative_response_table(irfm_bf['benef'], irfm_bf['full'], shock='i',
                          savepath=f'{tables}monetary_cumulative.tex', label='tab:monetary_cumulative')



# ---------------------------------------------------------------------------
# Robustness Tests
solve = lambda cal, **kw: solve_ss(hank_ss, cal, flows=flows, **kw)
build = lambda cal: irf_builder(hank, SteadyStateDict({**dyn, **cal}, internals=dyn.internals),
                                {**calibration, **cal}, unk_tax, tgt_tax, var_tax, wdw)


## Steady State
rob  = robustness(solve, calibration, warm_ss=ss)
rob |= bf_sizes(solve, calibration)
robustness_table(rob, savepath=f'{tables}robustness.tex')
bf_size_table(rob, savepath=f'{tables}robustness_bf.tex')


## Dynamics
dyn_rob = dyn_robustness(build, 'rstar', di)
dyn_rob_table(dyn_rob, savepath=f'{tables}dyn_robustness.tex')
plot_dyn_robustness(dyn_rob, savepath=f'{figures}dyn_robustness.png')



# ---------------------------------------------------------------------------
# Reload p6, p8 and p9 (interactive)
import importlib, code.p6_solve, code.p8_results, code.p9_robustness
importlib.reload(code.p6_solve)
importlib.reload(code.p8_results)
importlib.reload(code.p9_robustness)
for mod in (code.p6_solve, code.p8_results, code.p9_robustness):
    globals().update({k: v for k, v in vars(mod).items() if not k.startswith('_')})

from code.p6_solve import *
from code.p8_results import *
from code.p9_robustness import *

