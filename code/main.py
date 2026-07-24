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

# ---- Packages --------------------------------------------------------------
import random
import numpy as np
from sequence_jacobian import create_model

random.seed(20260415)


# Import parameters
from code.parameters import *
from code.other_blocks import *
from code.results import *          # B diverges in debt-dynamics
from code.household_block import hh, solve_ss, solve_dyn


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh, firm_formal, firm_informal, informal_wage,
                        nkpc_ss, union_ss, monetary, fiscal, mkt_clearing])

ss = solve_ss(hank_ss, calibration, unknowns, targets, verbose=True)


# Steady State Diagnostics
print_ss_summary(ss, ['Y', 'Y_I', 'C_GHH', 'C', 'beta_high', 'A', 'B',
                      'psi', 'w', 'w_I', 'Z', 'F', 'I', 'U', 'BF', 'L', 'Div',
                      'tau', 'asset_mkt', 'goods_mkt', 'labor_mkt', 'wage_nkpc'])


# No-BF Counterfactuals
ss_nobf = solve_ss(hank_ss, {**calibration, 'Tr': 0.0}, unknowns, targets, verbose=True)


# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh, firm_formal, firm_informal, informal_wage,
                     phillips_curve, monetary, fiscal, mkt_clearing])

dyn = hank.steady_state(ss)

# Verify Dyn is also a valid Steady State
for k in dyn.keys():
    assert np.all(np.isclose(dyn[k], ss[k], atol=1e-6)), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")


# ---------------------------------------------------------------------------
# Equilibrium Jacobians
T = 100
unknowns_tax  = ['Y', 'pi', 'w', 'tau_l']     # revenue-financed
unknowns_debt = ['Y', 'pi', 'w', 'B']         # debt-financed
targets_dyn   = ['goods_mkt', 'nkpc', 'wage_nkpc', 'gov_budget']
variables     = ['C', 'Y', 'L', 'I', 'U', 'BF', 'pi', 'w', 'r', 'i']


# IRFs:   AR(1) shock,      with   rho = 0.4,  size = 1%,  1 year antecipation
dTr0 = 0.01; rho_Tr = 0.40; k = 4
dTr  = dTr0 * rho_Tr ** np.arange(T)                # MIT Shock
dTr_ant  = np.concatenate([np.zeros(k), dTr[:-k]])  # Antecipated Shock: Perfect Foresight
dTr_perm = dTr0 * np.ones(T)                        # Permanent Shock


## Build IRFs
def build_irfs(unknowns_reg, variables, verbose=False):
    run = lambda shock, mv: solve_dyn(hank, ss, unknowns_reg, targets_dyn,
                                      shock, calibration, variables,
                                      moving=mv, verbose=verbose)
    full, insu, ant = run(dTr, True), run(dTr, False), run(dTr_ant, True)
    comp = {v: full[v] - insu[v] for v in variables}
    return dict(insu=insu, full=full, ant=ant, comp=comp)

G_hh     = hh.jacobian(dyn, inputs=['Tr', 'r'], T=T)
irf_tax  = build_irfs(unknowns_tax, variables)
irf_debt = build_irfs(unknowns_debt, variables+['B'])

irf_pe = {v: G_hh[v]['Tr'] @ dTr for v in variables if v in G_hh.outputs}


# ---------------------------------------------------------------------------
# Steady-State - Descriptive Statistics
compare_bf_ss(ss, ss_nobf, savepath='output/tables/ss_comparison.tex')
plot_descriptives(ss, ss_nobf, calibration, savepath='output/figures/bf_descript.png')


# Steady State - Distribution and Policy Functions
lorenz_scf_raw = np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=',')
plot_consumption_policy(ss, calibration, savepath='output/figures/consump_policy.png')
plot_wealth_distribution(ss, lorenz_scf_raw, savepath='output/figures/wealth_distribution.png')


# plot_bf_sweep(lambda cal: solve_ss(hank_ss, cal, unknowns, targets),
#               calibration, ss, span=0.2, nT=5, savepath='output/figures/bf_sweep.png')


# ---------------------------------------------------------------------------
# Dynamics - Partial Jacobians
plot_impc(G_hh, savepath='output/figures/impc.png')


# Dynamics - General Jacobians
plot_irf_financing(irf_tax['full'], irf_debt['full'],
                   savepath='output/figures/irf_financing.png')


plot_irf_decomposition(irf_debt['insu'], irf_debt['full'], irf_pe,
                       savepath='output/figures/irf_decomposition.png')


plot_irf({'MIT Shock': irf_debt['full'], 'Antecipated Shock': irf_debt['ant']},
         savepath='output/figures/irf_antecipated.png')


# Dynamics - Cumulative Response
cumulative_response_table(irf_debt['ant'], irf_debt['full'], var='C',
                          savepath='output/tables/response_consump.tex')

